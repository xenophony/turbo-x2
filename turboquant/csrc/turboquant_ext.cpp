/*
 * PyTorch C++ extension bindings for TurboQuant CUDA kernels.
 *
 * Exposes one function: turboquant_forward
 * Python calls this ONCE per layer — all groups handled internally in C++/CUDA.
 */

#include <torch/extension.h>

// Declared in turboquant_cuda_kernel.cu
torch::Tensor turboquant_forward_cuda(
    torch::Tensor x,
    torch::Tensor indices_packed,
    torch::Tensor codebook,
    torch::Tensor weight_norms,
    torch::Tensor group_signs,
    int group_size,
    int bit_width,
    bool use_hadamard
);

// Input validation + dispatch
torch::Tensor turboquant_forward(
    torch::Tensor x,
    torch::Tensor indices_packed,
    torch::Tensor codebook,
    torch::Tensor weight_norms,
    torch::Tensor group_signs,
    int group_size,
    int bit_width,
    bool use_hadamard
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(indices_packed.device().is_cuda(), "indices_packed must be a CUDA tensor");
    TORCH_CHECK(codebook.device().is_cuda(), "codebook must be a CUDA tensor");
    TORCH_CHECK(weight_norms.device().is_cuda(), "weight_norms must be a CUDA tensor");
    TORCH_CHECK(group_signs.device().is_cuda(), "group_signs must be a CUDA tensor");
    TORCH_CHECK(bit_width == 2 || bit_width == 4, "bit_width must be 2 or 4");

    return turboquant_forward_cuda(
        x, indices_packed, codebook, weight_norms, group_signs,
        group_size, bit_width, use_hadamard
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &turboquant_forward,
          "TurboQuant fused forward (CUDA)",
          py::arg("x"),
          py::arg("indices_packed"),
          py::arg("codebook"),
          py::arg("weight_norms"),
          py::arg("group_signs"),
          py::arg("group_size"),
          py::arg("bit_width"),
          py::arg("use_hadamard"));
}
