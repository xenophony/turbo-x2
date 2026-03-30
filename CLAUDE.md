# TurboQuant 2-bit Weight Quantization — Implementation Spec

## Goal

Implement 2-bit TurboQuant weight quantization for HuggingFace transformer models,
enabling large models (32B) to run on consumer GPUs (RTX 3060, 12GB VRAM).

Validate end-to-end locally on a small model (Qwen2.5-0.5B) before scaling to
larger models on Vast.ai.

---

## Background

### The Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
- Zandieh et al., ICLR 2026
- https://arxiv.org/abs/2504.19874
- Official scope: KV cache compression and vector search
- Key claim: near-optimal distortion at **any bit width** (proven near theoretical lower bound)

### The Two-Stage Method (applied to weights by community)

1. **PolarQuant / Rotation stage**: Randomly rotate each weight row using an orthogonal
   matrix Pi. After rotation, coordinates follow a concentrated Beta distribution,
   making them amenable to near-optimal scalar quantization per coordinate.

2. **Lloyd-Max scalar quantization**: For each rotated coordinate, assign to the
   nearest centroid from a precomputed Lloyd-Max codebook optimized for the
   resulting distribution. At 2-bit this is 4 centroids; at 3-bit, 8 centroids.

Why this beats standard quantization: standard methods (GPTQ, AWQ) quantize in the
original weight space where the distribution is irregular. Rotation first induces a
known, predictable distribution, making the quantizer near-optimal by theory.

### Community Implementation

**cksac/turboquant-model** (released March 27, 2026, MIT license)
- Implements TurboQuant for model weights as a drop-in `nn.Linear` replacement
- Currently supports: 4-bit single-pass, residual combinations (3+2, 4+2, 4+4)
- **Gap**: packed storage and inference kernels are hardcoded to 4-bit
- The simulation path (`turboquant_quantize`) already supports any bit width

---

## What Exists in the Upstream Repo

### `codebook.py` — already supports any bit width

```python
def get_codebook(bit_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get precomputed Lloyd-Max codebook for given bit-width.
    Works for any bit_width: 1, 2, 3, 4, ...
    Returns centroids (2^bit_width,) and boundaries (2^bit_width - 1,)
    """
    n_levels = 2**bit_width
    centroids, boundaries = _compute_lloyd_max_gaussian(n_levels)
    ...
```

`get_codebook(2)` gives 4 centroids. Already works. No changes needed.

### `quantize.py` — simulation works, packing is 4-bit only

```python
# This already works for any bit_width including 2:
def turboquant_quantize(W, bit_width=4, group_size=None, seed=42, rotation="qr"):
    centroids, boundaries = get_codebook(bit_width)  # works for bit_width=2
    # ... rotate, quantize, dequantize ...
    return W_approx  # float32 approximation

# This is BROKEN for 2-bit — hardcoded assertion:
def turboquant_quantize_packed(W, bit_width=4, ...):
    assert bit_width == 4, "Packed format supports 4-bit only"  # <-- FIX THIS
    # Uses pack_4bit which packs 2 values per byte (nibbles)
```

Existing 4-bit packing for reference:
```python
def pack_4bit(indices):  # indices in [0, 15], 2 per byte
    lo = indices[..., 0::2].to(torch.uint8)
    hi = indices[..., 1::2].to(torch.uint8)
    return lo | (hi << 4)

def unpack_4bit(packed, N):
    lo = (packed & 0x0F).to(torch.int32)
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    result = torch.stack([lo, hi], dim=-1)
    return result.reshape(*packed.shape[:-1], N)
```

### `module.py` — architecture is generic, implementation is 4-bit

The `TurboQuantLinear` class already computes the correct packed dimension:
```python
pack_factor = 8 // bit_width   # For 2-bit: pack_factor=4 (4 values per byte)
packed_dim = math.ceil(in_features / pack_factor)
```

But `_get_indices()` calls `unpack_4bit` unconditionally — needs to dispatch on `bit_width`.

The forward pass and residual logic are otherwise correct for any bit width.

---

## What Needs to Be Built

### 1. 2-bit (and generic n-bit) packing

```python
def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices (0-3) into uint8, 4 per byte."""
    assert indices.shape[-1] % 4 == 0
    b0 = indices[..., 0::4].to(torch.uint8)
    b1 = indices[..., 1::4].to(torch.uint8)
    b2 = indices[..., 2::4].to(torch.uint8)
    b3 = indices[..., 3::4].to(torch.uint8)
    return b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)

def unpack_2bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """Unpack uint8 -> 2-bit indices."""
    b0 = (packed & 0x03).to(torch.int32)
    b1 = ((packed >> 2) & 0x03).to(torch.int32)
    b2 = ((packed >> 4) & 0x03).to(torch.int32)
    b3 = ((packed >> 6) & 0x03).to(torch.int32)
    result = torch.stack([b0, b1, b2, b3], dim=-1)
    return result.reshape(*packed.shape[:-1], -1)[..., :N]
```

Build a generic dispatcher:
```python
def pack_bits(indices, bit_width):
    if bit_width == 2: return pack_2bit(indices)
    if bit_width == 4: return pack_4bit(indices)
    raise ValueError(f"Unsupported bit_width={bit_width}")

def unpack_bits(packed, N, bit_width):
    if bit_width == 2: return unpack_2bit(packed, N)
    if bit_width == 4: return unpack_4bit(packed, N)
    raise ValueError(f"Unsupported bit_width={bit_width}")
```

### 2. Fix `turboquant_quantize_packed` to support 2-bit

Remove the `assert bit_width == 4` and replace `pack_4bit` with `pack_bits(indices, bit_width)`.
Padding logic changes: for 2-bit, pad to multiple of 4 (not 2).

### 3. Fix `TurboQuantLinear._get_indices()` dispatch

```python
def _get_indices(self) -> torch.Tensor:
    if self._cached_indices is None:
        self._cached_indices = unpack_bits(
            self.indices_packed, self.in_features, self.bit_width
        )
    return self._cached_indices
```

### 4. Fix `merge_passes` packing call

In `module.py`, `merge_passes()` calls `pack_4bit` directly — replace with `pack_bits`.

### 5. Quantization script

```python
# quantize.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path

def quantize_model(model_id: str, output_dir: str, bit_width: int = 2):
    print(f"Loading {model_id} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Quantizing to {bit_width}-bit...")
    replace_linear_layers(model, bit_width=bit_width)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")
```

`replace_linear_layers` walks `model.named_modules()`, finds `nn.Linear` instances,
quantizes each weight tensor, and swaps in a `TurboQuantLinear`.

Skip: `lm_head`, embedding layers (these are lookup tables, not matmuls).

### 6. Inference script

```python
# inference.py
from transformers import AutoTokenizer
import torch

def load_and_generate(model_dir: str, prompt: str, max_new_tokens: int = 200):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_turboquant_model(model_dir)  # custom loader
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.2 | Core tensor ops, packing, matmul |
| `transformers` | ≥4.40 | Load HuggingFace models, tokenizers |
| `scipy` | any | Lloyd-Max codebook computation (scipy.stats.norm) |
| `numpy` | any | Codebook precomputation |
| `safetensors` | any | Efficient weight serialization |
| `accelerate` | any | `device_map="auto"` for large model loading |
| `triton` | optional | Fused inference kernel (skip for initial version) |

Do NOT add the upstream `turboquant-model` as a dependency — copy and modify the
relevant source files directly. The upstream is too new and too tightly coupled to 4-bit.

Install:
```bash
pip install torch transformers scipy numpy safetensors accelerate
```

---

## Repo Structure

```
turboquant-2bit/
├── turboquant/
│   ├── __init__.py
│   ├── codebook.py     ← copy from upstream, no changes needed
│   ├── rotation.py     ← copy from upstream, no changes needed
│   ├── packing.py      ← NEW: pack_2bit, unpack_2bit, pack_bits, unpack_bits
│   ├── quantize.py     ← modified: remove assert, use pack_bits
│   ├── module.py       ← modified: dispatch on bit_width in _get_indices, merge_passes
│   └── model.py        ← NEW: replace_linear_layers, save/load quantized model
├── quantize.py         ← CLI: python quantize.py --model <hf_id> --bits 2 --output ./quantized
├── inference.py        ← CLI: python inference.py --model ./quantized --prompt "hello"
├── eval.py             ← quality comparison: perplexity of original vs quantized
├── requirements.txt
└── README.md
```

---

## Test Plan

### Phase 1 — Local (RTX 3060, no Vast.ai needed)

**Model**: `Qwen/Qwen2.5-0.5B-Instruct` (~1GB download, ~125MB at 2-bit)

```bash
# Step 1: quantize
python quantize.py --model Qwen/Qwen2.5-0.5B-Instruct --bits 2 --output ./quantized/qwen-0.5b-2bit

# Step 2: check size
du -sh ./quantized/qwen-0.5b-2bit/

# Step 3: run inference
python inference.py --model ./quantized/qwen-0.5b-2bit --prompt "The capital of France is"

# Step 4: compare quality
python eval.py --original Qwen/Qwen2.5-0.5B-Instruct --quantized ./quantized/qwen-0.5b-2bit
```

`eval.py` should compute and print:
- Perplexity on a fixed text sample (e.g. first 1000 tokens of wikitext)
- Side-by-side generation on 5 prompts
- Memory usage before/after

Acceptance criteria:
- Quantized model loads and generates without errors
- Perplexity degradation < 20% vs original (2-bit will have some quality loss — that's expected)
- File size ≈ original_size / 8 (2-bit vs 16-bit)

### Phase 2 — Scale up on Vast.ai

Once Phase 1 passes:
- Rent instance with ≥48GB VRAM (A6000 or 2x A100)
- Run `quantize.py --model Qwen/Qwen2.5-32B-Instruct --bits 2`
- Download quantized model (~8GB)
- Run locally on 3060

---

## Memory Estimates

| Model | bf16 size | 2-bit size | Fits on 3060 (12GB)? |
|-------|-----------|------------|----------------------|
| Qwen2.5-0.5B | ~1GB | ~125MB | Yes (trivially) |
| Qwen2.5-1.5B | ~3GB | ~375MB | Yes |
| Qwen2.5-7B | ~14GB | ~1.75GB | Yes |
| Qwen2.5-14B | ~28GB | ~3.5GB | Yes, 8GB spare for KV |
| Qwen2.5-32B | ~64GB | ~8GB | Yes, 4GB spare for KV |
| Qwen2.5-72B | ~144GB | ~18GB | No (needs CPU offload) |

Note: these are weight-only estimates. KV cache adds ~0.5–2GB depending on context length.

---

## Key Implementation Notes

### Rotation matrix generation

The upstream uses two methods — prefer Hadamard for speed:
```python
# Fast Walsh-Hadamard (preferred, O(n log n))
from turboquant.rotation import hadamard_rotate, hadamard_rotate_inverse

# QR decomposition (slower, O(n^2), but exact orthogonal)
from turboquant.rotation import generate_rotation_matrix
```

Use `rotation="hadamard"` as default. It's faster and the upstream proves it works
as well as QR for this application.

### Seed consistency

The same seed must be used at quantization time and inference time — it determines
the rotation matrix Pi. Store the seed in the saved model config. The upstream
stores it as `_rotation_seed` on the module.

### Skipping non-linear layers

Only quantize `nn.Linear` layers. Skip:
- `model.embed_tokens` (embedding lookup, not matmul)
- `lm_head` (output projection — quantizing this hurts output token probabilities)
- Any layer with `in_features < 64` (too small to benefit)

### Layer-by-layer memory management

For large models on Vast.ai, load in CPU and move each layer to GPU for quantization:
```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and should_quantize(name):
        module.to("cuda")
        # quantize
        module.to("cpu")
        torch.cuda.empty_cache()
```

This lets you quantize models larger than GPU VRAM.

---

## Quality Expectations at 2-bit

2-bit is aggressive. Realistic expectations:
- Simple factual recall: mostly intact
- Reasoning chains: noticeably degraded vs 4-bit
- Fluency: generally maintained
- Perplexity: expect ~15–35% increase over bf16

TurboQuant's rotation trick should do better than naive 2-bit (e.g. standard GPTQ
at 2-bit is usually unusable). The Lloyd-Max codebook is near-optimal for the
rotated distribution. But 2-bit is still 2-bit — manage expectations accordingly.

For serious agentic use, 3-bit is probably the practical floor. The same code
supports 3-bit with `--bits 3` (8 centroids, ~2x the quality of 2-bit).

---

## Starting Point for the Agent

Read these files from upstream first before writing anything:
- https://github.com/cksac/turboquant-model/blob/main/src/turboquant_model/codebook.py
- https://github.com/cksac/turboquant-model/blob/main/src/turboquant_model/rotation.py
- https://github.com/cksac/turboquant-model/blob/main/src/turboquant_model/quantize.py
- https://github.com/cksac/turboquant-model/blob/main/src/turboquant_model/module.py
- https://github.com/cksac/turboquant-model/blob/main/src/turboquant_model/residual.py

Do not modify the upstream repo. Copy files into `turboquant/` and modify from there.

Implement in this order:
1. `turboquant/packing.py` — write and test `pack_2bit` / `unpack_2bit` first
2. `turboquant/codebook.py` + `turboquant/rotation.py` — copy verbatim
3. `turboquant/quantize.py` — copy and fix the `assert bit_width == 4`
4. `turboquant/module.py` — copy and fix `_get_indices`, `merge_passes`
5. `turboquant/model.py` — write `replace_linear_layers`, save/load
6. `quantize.py` CLI
7. `inference.py` CLI
8. `eval.py` quality comparison

Test after each step. Do not proceed to the next step if the current one fails.
