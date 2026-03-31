/*
 * TurboQuant fused CUDA kernels
 *
 * Two kernels:
 *   1. hadamard_rotate_kernel — in-place sign flip + Fast Walsh-Hadamard Transform
 *   2. lut_2bit_matmul_kernel — LUT-based 2-bit dequant + matmul (the "Doom trick")
 *
 * Combined, these handle one full TurboQuantLinear forward per layer
 * with zero Python overhead in the inner loop.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Hadamard rotation kernel
// ---------------------------------------------------------------------------

// In-place: x[b, :] = FWHT(x[b, :] * signs) / sqrt(K)
// Requires K to be power of 2 and K <= 1024 (one block per batch row)
__global__ void hadamard_rotate_kernel(
    float* __restrict__ x,          // (B, K) — modified in-place
    const float* __restrict__ signs, // (K,) random ±1
    const int B,
    const int K
) {
    const int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float shared[];
    float* s = shared;  // K floats in shared memory

    // Load x[b, :] into shared memory, apply signs
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        s[k] = x[b * K + k] * signs[k];
    }
    __syncthreads();

    // Fast Walsh-Hadamard Transform (butterfly)
    for (int h = 1; h < K; h *= 2) {
        for (int k = threadIdx.x; k < K / 2; k += blockDim.x) {
            int i = (k / h) * (2 * h) + (k % h);
            int j = i + h;
            float a = s[i];
            float bv = s[j];
            s[i] = a + bv;
            s[j] = a - bv;
        }
        __syncthreads();
    }

    // Write back normalized
    float inv_sqrt_k = rsqrtf((float)K);
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        x[b * K + k] = s[k] * inv_sqrt_k;
    }
}

void hadamard_rotate_cuda(torch::Tensor x, torch::Tensor signs) {
    const int B = x.size(0);
    const int K = x.size(1);

    const int threads = min(K / 2, 1024);
    const int shared_mem = K * sizeof(float);

    hadamard_rotate_kernel<<<B, threads, shared_mem>>>(
        x.data_ptr<float>(),
        signs.data_ptr<float>(),
        B, K
    );
}


// ---------------------------------------------------------------------------
// LUT 2-bit matmul kernel
// ---------------------------------------------------------------------------

// For each output element (b, n):
//   output[b][n] = norms_scaled[n] * sum_k table[idx[n][k]][b][k]
// where table[c][b][k] = x_rot[b][k] * codebook[c]
//
// We decompose into: for each centroid c, compute (x*c) @ mask_c.T
// This is 4 "binary matmuls" — the mask is 0/1.

#define TILE_N 32
#define TILE_K 128
#define N_CENTROIDS 4

__global__ void lut_2bit_matmul_kernel(
    const float* __restrict__ x_rot,        // (B, K)
    const uint8_t* __restrict__ indices_packed, // (N, K/4)
    const float* __restrict__ codebook,     // (4,)
    const float* __restrict__ norms_scaled, // (N,)
    float* __restrict__ output,             // (B, N)
    const int B,
    const int N,
    const int K
) {
    const int packed_K = K / 4;
    const int b = blockIdx.x;
    const int n_start = blockIdx.y * TILE_N;

    if (b >= B) return;

    // Load codebook into registers (4 values)
    float cb[N_CENTROIDS];
    for (int c = 0; c < N_CENTROIDS; c++) {
        cb[c] = codebook[c];
    }

    // Each thread handles one or more output neurons
    for (int tn = threadIdx.x; tn < TILE_N && (n_start + tn) < N; tn += blockDim.x) {
        const int n = n_start + tn;
        float acc = 0.0f;

        // Stream through K dimension, 4 elements at a time (one packed byte)
        for (int pk = 0; pk < packed_K; pk++) {
            uint8_t packed = indices_packed[n * packed_K + pk];
            int base_k = pk * 4;

            // Unpack 4 indices and accumulate
            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                int idx = (packed >> (sub * 2)) & 0x03;
                int k = base_k + sub;
                if (k < K) {
                    acc += x_rot[b * K + k] * cb[idx];
                }
            }
        }

        // Apply pre-scaled norm and accumulate to output
        atomicAdd(&output[b * N + n], acc * norms_scaled[n]);
    }
}


void lut_2bit_matmul_cuda(
    torch::Tensor x_rot,           // (B, K) float32
    torch::Tensor indices_packed,   // (N, K/4) uint8
    torch::Tensor codebook,         // (4,) float32
    torch::Tensor norms_scaled,     // (N,) float32
    torch::Tensor output,           // (B, N) float32 — accumulated into
    int K
) {
    const int B = x_rot.size(0);
    const int N = indices_packed.size(0);

    dim3 grid(B, (N + TILE_N - 1) / TILE_N);
    int threads = min(TILE_N, 256);

    lut_2bit_matmul_kernel<<<grid, threads>>>(
        x_rot.data_ptr<float>(),
        indices_packed.data_ptr<uint8_t>(),
        codebook.data_ptr<float>(),
        norms_scaled.data_ptr<float>(),
        output.data_ptr<float>(),
        B, N, K
    );
}


// ---------------------------------------------------------------------------
// LUT 4-bit matmul kernel
// ---------------------------------------------------------------------------

#define N_CENTROIDS_4BIT 16

__global__ void lut_4bit_matmul_kernel(
    const float* __restrict__ x_rot,
    const uint8_t* __restrict__ indices_packed, // (N, K/2)
    const float* __restrict__ codebook,     // (16,)
    const float* __restrict__ norms_scaled, // (N,)
    float* __restrict__ output,             // (B, N)
    const int B,
    const int N,
    const int K
) {
    const int packed_K = K / 2;
    const int b = blockIdx.x;
    const int n_start = blockIdx.y * TILE_N;

    if (b >= B) return;

    // Load codebook into registers (16 values)
    float cb[N_CENTROIDS_4BIT];
    for (int c = 0; c < N_CENTROIDS_4BIT; c++) {
        cb[c] = codebook[c];
    }

    for (int tn = threadIdx.x; tn < TILE_N && (n_start + tn) < N; tn += blockDim.x) {
        const int n = n_start + tn;
        float acc = 0.0f;

        for (int pk = 0; pk < packed_K; pk++) {
            uint8_t packed = indices_packed[n * packed_K + pk];
            int base_k = pk * 2;

            int idx_lo = packed & 0x0F;
            int idx_hi = (packed >> 4) & 0x0F;

            acc += x_rot[b * K + base_k] * cb[idx_lo];
            if (base_k + 1 < K) {
                acc += x_rot[b * K + base_k + 1] * cb[idx_hi];
            }
        }

        atomicAdd(&output[b * N + n], acc * norms_scaled[n]);
    }
}


void lut_4bit_matmul_cuda(
    torch::Tensor x_rot,
    torch::Tensor indices_packed,
    torch::Tensor codebook,
    torch::Tensor norms_scaled,
    torch::Tensor output,
    int K
) {
    const int B = x_rot.size(0);
    const int N = indices_packed.size(0);

    dim3 grid(B, (N + TILE_N - 1) / TILE_N);
    int threads = min(TILE_N, 256);

    lut_4bit_matmul_kernel<<<grid, threads>>>(
        x_rot.data_ptr<float>(),
        indices_packed.data_ptr<uint8_t>(),
        codebook.data_ptr<float>(),
        norms_scaled.data_ptr<float>(),
        output.data_ptr<float>(),
        B, N, K
    );
}


// ---------------------------------------------------------------------------
// Full layer forward — handles all groups, one C++ call per layer
// ---------------------------------------------------------------------------

torch::Tensor turboquant_forward_cuda(
    torch::Tensor x,               // (B, in_features) bf16/f32
    torch::Tensor indices_packed,   // (out_features, packed_in) uint8
    torch::Tensor codebook,         // (n_levels,) f32
    torch::Tensor weight_norms,     // (out_features,) or (out_features, n_groups) f32
    torch::Tensor group_signs,      // (n_groups, group_size) f32 — precomputed ±1 signs
    int group_size,
    int bit_width,
    bool use_hadamard
) {
    const int B = x.size(0);
    const int in_features = x.size(1);
    const int out_features = indices_packed.size(0);
    const int n_groups = (in_features + group_size - 1) / group_size;
    const float scale = sqrtf((float)group_size);
    const int pack_factor = 8 / bit_width;

    // Convert input to float32 for computation
    auto x_f32 = x.to(torch::kFloat32);

    // Output accumulator
    auto output = torch::zeros({B, out_features},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    for (int g = 0; g < n_groups; g++) {
        int g_start = g * group_size;
        int g_end = std::min(g_start + group_size, in_features);
        int g_dim = g_end - g_start;

        // 1. Extract input slice for this group
        auto x_slice = x_f32.slice(1, g_start, g_end).contiguous();

        // 2. Hadamard rotation (in-place on x_slice)
        if (use_hadamard) {
            auto signs = group_signs[g].slice(0, 0, g_dim);
            hadamard_rotate_cuda(x_slice, signs);
        }
        // QR rotation would go here but we always use hadamard with group_size=1024

        // 3. Get norms for this group, pre-scale
        torch::Tensor norms_g;
        if (weight_norms.dim() == 1) {
            norms_g = weight_norms / scale;
        } else {
            norms_g = weight_norms.select(1, g) / scale;
        }

        // 4. Extract packed indices for this group
        int packed_g_start = g_start / pack_factor;
        int packed_g_end = (g_end + pack_factor - 1) / pack_factor;
        auto packed_g = indices_packed.slice(1, packed_g_start, packed_g_end).contiguous();

        // 5. LUT matmul (accumulates into output)
        if (bit_width == 2) {
            lut_2bit_matmul_cuda(x_slice, packed_g, codebook, norms_g, output, g_dim);
        } else {
            lut_4bit_matmul_cuda(x_slice, packed_g, codebook, norms_g, output, g_dim);
        }
    }

    return output;
}
