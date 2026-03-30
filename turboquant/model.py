"""Model-level utilities: quantize, save, and load HuggingFace transformer models."""

from __future__ import annotations

import json
import math
import gc
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from turboquant.codebook import get_codebook
from turboquant.module import TurboQuantLinear, TurboQuantEmbedding
from turboquant.quantize import turboquant_quantize_packed


# ---------------------------------------------------------------------------
# Layer filtering
# ---------------------------------------------------------------------------

# lm_head is handled specially (tied to embedding or quantized as linear)
_SKIP_PATTERNS_LINEAR = {"lm_head", "embed_tokens", "embed_positions", "wte", "wpe"}
_MIN_FEATURES = 64


def should_quantize_linear(name: str, module: nn.Linear) -> bool:
    """Decide whether a Linear layer should be quantized."""
    for pat in _SKIP_PATTERNS_LINEAR:
        if pat in name:
            return False
    if module.in_features < _MIN_FEATURES or module.out_features < _MIN_FEATURES:
        return False
    return True


# ---------------------------------------------------------------------------
# Replace nn.Linear with TurboQuantLinear
# ---------------------------------------------------------------------------


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    """Set a submodule by dotted name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _quantize_embedding(
    module: nn.Embedding,
    bit_width: int,
    group_size: Optional[int],
    seed: int,
    rotation: str,
    device: str,
) -> TurboQuantEmbedding:
    """Quantize an nn.Embedding into a TurboQuantEmbedding."""
    W = module.weight.data
    W_dev = W.to(device)

    packed = turboquant_quantize_packed(
        W_dev, bit_width=bit_width, group_size=group_size,
        seed=seed, rotation=rotation,
    )

    num_emb, emb_dim = packed["shape"]
    tq_emb = TurboQuantEmbedding(
        num_embeddings=num_emb,
        embedding_dim=emb_dim,
        bit_width=bit_width,
        group_size=packed["group_size"],
        rotation=rotation,
        padding_idx=module.padding_idx,
    )
    tq_emb.indices_packed.copy_(packed["indices_packed"])
    tq_emb.codebook.copy_(packed["codebook"])
    tq_emb.weight_norms.copy_(packed["norms"])
    tq_emb.set_rotation(seed)

    del W_dev, packed
    return tq_emb.cpu()


@torch.no_grad()
def replace_linear_layers(
    model: nn.Module,
    bit_width: int = 2,
    group_size: Optional[int] = None,
    seed: int = 42,
    rotation: str = "hadamard",
    device: Optional[str] = None,
    quantize_embeddings: bool = True,
    embedding_bit_width: Optional[int] = None,
) -> nn.Module:
    """Walk model, quantize eligible nn.Linear and nn.Embedding layers.

    Handles weight tying: if lm_head shares weights with the embedding,
    lm_head is replaced with a TurboQuantLinear sharing the same packed data.

    Args:
        model: HuggingFace model
        bit_width: bits per weight coordinate (2 or 4)
        group_size: quantization group size (None = full row)
        seed: rotation seed
        rotation: "hadamard" or "qr"
        device: device for quantization computation (None = auto-detect)
        quantize_embeddings: if True, also quantize nn.Embedding layers

    Returns:
        The modified model (in-place)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Quantize embeddings ---
    embeddings_to_replace = []
    if quantize_embeddings:
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and module.embedding_dim >= _MIN_FEATURES:
                embeddings_to_replace.append(name)

    emb_bits = embedding_bit_width or bit_width
    if embeddings_to_replace:
        print(f"Quantizing {len(embeddings_to_replace)} embedding(s) to {emb_bits}-bit...")
        for name in embeddings_to_replace:
            parts = name.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)

            tq_emb = _quantize_embedding(module, emb_bits, group_size, seed, rotation, device)
            _set_module(model, name, tq_emb)
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  {name} ({module.num_embeddings}x{module.embedding_dim})")

    # --- Handle lm_head tied to embedding ---
    # Check if lm_head exists and was tied to the embedding
    config = getattr(model, "config", None)
    tied = config and getattr(config, "tie_word_embeddings", False)

    if tied and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        # Replace lm_head with a module that dequantizes from the embedding
        # Find the quantized embedding
        emb_module = None
        for name in embeddings_to_replace:
            parts = name.split(".")
            m = model
            for part in parts:
                m = getattr(m, part)
            if isinstance(m, TurboQuantEmbedding):
                emb_module = m
                break

        if emb_module is not None:
            lm_head = _make_tq_linear_from_embedding(emb_module)
            model.lm_head = lm_head
            print(f"  lm_head → TurboQuantLinear tied to quantized embedding")

    # --- Quantize linear layers ---
    layers_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_quantize_linear(name, module):
            layers_to_replace.append(name)

    print(f"Quantizing {len(layers_to_replace)} linear layers to {bit_width}-bit...")

    for i, name in enumerate(layers_to_replace):
        parts = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)

        W = module.weight.data
        has_bias = module.bias is not None
        bias_data = module.bias.data if has_bias else None

        W_dev = W.to(device)

        packed = turboquant_quantize_packed(
            W_dev, bit_width=bit_width, group_size=group_size,
            seed=seed, rotation=rotation,
        )

        M, N = packed["shape"]
        tq = TurboQuantLinear(
            in_features=N,
            out_features=M,
            bias=has_bias,
            bit_width=bit_width,
            group_size=packed["group_size"],
            rotation=rotation,
        )
        tq.indices_packed.copy_(packed["indices_packed"])
        tq.codebook.copy_(packed["codebook"])
        tq.weight_norms.copy_(packed["norms"])
        tq.set_rotation(seed)
        if has_bias:
            tq.bias.copy_(bias_data)

        _set_module(model, name, tq.cpu())

        del W_dev, packed
        if device == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or (i + 1) == len(layers_to_replace):
            print(f"  [{i+1}/{len(layers_to_replace)}] {name}")

    return model


def _make_tq_linear_from_embedding(emb: TurboQuantEmbedding) -> TurboQuantLinear:
    """Create a TurboQuantLinear that shares quantized data with an embedding.

    The embedding matrix (num_embeddings × embedding_dim) IS the linear weight
    (out_features=num_embeddings, in_features=embedding_dim). This lets lm_head
    use the efficient on-the-fly rotation trick instead of full dequantization.
    """
    tq = TurboQuantLinear(
        in_features=emb.embedding_dim,
        out_features=emb.num_embeddings,
        bias=False,
        bit_width=emb.bit_width,
        group_size=emb.group_size,
        rotation=emb.rotation,
    )
    # Share buffers (no extra memory)
    tq.indices_packed = emb.indices_packed
    tq.weight_norms = emb.weight_norms
    tq.codebook = emb.codebook
    tq.set_rotation(emb._rotation_seed)
    return tq


# ---------------------------------------------------------------------------
# Save / Load quantized model
# ---------------------------------------------------------------------------


def save_quantized_model(model: nn.Module, output_dir: str, tokenizer=None):
    """Save a quantized model to disk.

    Saves:
      - model.safetensors (or model.bin): all state_dict tensors
      - turboquant_config.json: metadata for each TurboQuantLinear layer
      - tokenizer files (if tokenizer provided)
    """
    from safetensors.torch import save_model

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect TurboQuantLinear and TurboQuantEmbedding metadata
    tq_config = {}
    for name, module in model.named_modules():
        if isinstance(module, TurboQuantLinear):
            tq_config[name] = {
                "type": "linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bit_width": module.bit_width,
                "group_size": module.group_size,
                "rotation": module.rotation,
                "seed": module._rotation_seed,
                "has_bias": module.bias is not None,
                "has_residual": module.has_residual,
            }
        elif isinstance(module, TurboQuantEmbedding):
            tq_config[name] = {
                "type": "embedding",
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim,
                "bit_width": module.bit_width,
                "group_size": module.group_size,
                "rotation": module.rotation,
                "seed": module._rotation_seed,
                "padding_idx": module.padding_idx,
            }
    # Detect tied lm_head (TurboQuantLinear sharing buffers with embedding)
    emb_names = [n for n, cfg in tq_config.items() if cfg.get("type") == "embedding"]
    for name, module in model.named_modules():
        if (name == "lm_head" and isinstance(module, TurboQuantLinear)
                and name not in tq_config and emb_names):
            tq_config[name] = {
                "type": "tied_lm_head",
                "tied_to": emb_names[0],
            }

    # Save config
    with open(output_path / "turboquant_config.json", "w") as f:
        json.dump(tq_config, f, indent=2)

    # Save state dict via safetensors (handles tied weights automatically)
    save_model(model, output_path / "model.safetensors")

    # Save the original model config (for loading the base architecture)
    if hasattr(model, "config"):
        model.config.save_pretrained(output_dir)

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Print size info
    model_file = output_path / "model.safetensors"
    size_mb = model_file.stat().st_size / 1024**2
    print(f"Saved quantized model to {output_dir} ({size_mb:.1f} MB)")


def load_quantized_model(
    model_dir: str,
    device: Optional[str] = None,
) -> nn.Module:
    """Load a quantized model from disk.

    Only needs the quantized model directory — does NOT download the
    original full-precision model. Creates the architecture from config,
    replaces quantized layers with TQ modules, and loads all weights
    from the saved safetensors file.

    Args:
        model_dir: directory containing model.safetensors and turboquant_config.json
        device: target device (None = auto)

    Returns:
        The loaded model with TurboQuantLinear layers
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = Path(model_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load turboquant config
    with open(model_path / "turboquant_config.json") as f:
        tq_config = json.load(f)

    # Create model on meta device (zero memory), replace quantized layers
    # with TQ modules (on CPU), then materialize remaining meta tensors.
    # This avoids allocating ~28GB of throwaway linear weights for 14B models.
    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # Replace quantized layers with TQ modules (created on CPU, not meta)
    emb_modules = {}
    for name, layer_cfg in tq_config.items():
        ltype = layer_cfg.get("type", "linear")

        if ltype == "linear":
            tq = TurboQuantLinear(
                in_features=layer_cfg["in_features"],
                out_features=layer_cfg["out_features"],
                bias=layer_cfg["has_bias"],
                bit_width=layer_cfg["bit_width"],
                group_size=layer_cfg["group_size"],
                rotation=layer_cfg["rotation"],
                device="cpu",
            )
            tq.set_rotation(layer_cfg["seed"])
            _set_module(model, name, tq)

        elif ltype == "embedding":
            tq_emb = TurboQuantEmbedding(
                num_embeddings=layer_cfg["num_embeddings"],
                embedding_dim=layer_cfg["embedding_dim"],
                bit_width=layer_cfg["bit_width"],
                group_size=layer_cfg["group_size"],
                rotation=layer_cfg["rotation"],
                padding_idx=layer_cfg.get("padding_idx"),
                device="cpu",
            )
            tq_emb.set_rotation(layer_cfg["seed"])
            _set_module(model, name, tq_emb)
            emb_modules[name] = tq_emb

        elif ltype == "tied_lm_head":
            pass  # handled in second pass

    # Second pass: tied lm_head (needs embedding module to exist first)
    for name, layer_cfg in tq_config.items():
        if layer_cfg.get("type") == "tied_lm_head":
            tied_to = layer_cfg.get("tied_to")
            emb = emb_modules.get(tied_to) or next(iter(emb_modules.values()), None)
            if emb is not None:
                _set_module(model, name, _make_tq_linear_from_embedding(emb))

    # Materialize any remaining meta modules on CPU.
    # TQ modules are already on CPU; only non-quantized leftovers need conversion.
    # Walk modules and convert any that still have meta tensors.
    for name, module in list(model.named_modules()):
        if isinstance(module, (TurboQuantLinear, TurboQuantEmbedding)):
            continue  # already on CPU
        has_meta = any(
            (p.device == torch.device("meta"))
            for p in list(module.parameters(recurse=False))
        ) or any(
            (b is not None and b.device == torch.device("meta"))
            for b in list(module.buffers(recurse=False))
        )
        if has_meta:
            # Recreate this module's tensors on CPU
            for pname, param in list(module.named_parameters(recurse=False)):
                if param.device == torch.device("meta"):
                    new_param = nn.Parameter(
                        torch.empty(param.shape, dtype=param.dtype, device="cpu"))
                    module.register_parameter(pname, new_param)
            for bname, buf in list(module.named_buffers(recurse=False)):
                if buf is not None and buf.device == torch.device("meta"):
                    module.register_buffer(bname, torch.empty(
                        buf.shape, dtype=buf.dtype, device="cpu"))

    # Recompute rotary embeddings (to_empty/meta zeros inv_freq)
    for name, module in model.named_modules():
        if hasattr(module, "rotary_emb"):
            rotary_cls = type(module.rotary_emb)
            try:
                module.rotary_emb = rotary_cls(config=config, device="cpu")
            except Exception:
                pass  # not all models have the same rotary API

    # Load ALL weights from our safetensors file (TQ buffers + layernorms + etc.)
    state_dict = load_file(model_path / "model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    # Restore tied/shared weights after load.
    # safetensors deduplicates shared buffers: when lm_head and embed_tokens share
    # the same packed data, only one copy is saved. Re-share after loading.
    if hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        lm = model.lm_head
        emb = model.model.embed_tokens
        if isinstance(lm, TurboQuantLinear) and isinstance(emb, TurboQuantEmbedding):
            # lm_head got the data, embedding is empty — share buffers
            emb.indices_packed = lm.indices_packed
            emb.codebook = lm.codebook
            emb.weight_norms = lm.weight_norms
        elif isinstance(emb, TurboQuantEmbedding) and isinstance(lm, nn.Linear):
            # lm_head wasn't quantized but embedding was — shouldn't happen, but handle it
            pass
        elif isinstance(emb, nn.Embedding) and isinstance(lm, nn.Linear):
            # Neither quantized — restore standard weight tying
            if hasattr(config, "tie_word_embeddings") and config.tie_word_embeddings:
                emb.weight = lm.weight

    model = model.to(device)
    return model


@torch.no_grad()
def decompress_model(model: nn.Module) -> nn.Module:
    """Decompress all TurboQuant layers back to standard nn.Linear / nn.Embedding.

    Dequantizes all weights once at load time so inference runs at native speed.
    Uses more VRAM than packed format but eliminates on-the-fly dequantization.

    For 14B at 2-bit: ~4GB decompressed (bf16) vs ~2.3GB packed.
    Still fits on a 3060 (12GB) with room for KV cache.

    Args:
        model: model with TurboQuantLinear / TurboQuantEmbedding layers

    Returns:
        The modified model with standard nn.Linear / nn.Embedding layers (in-place)
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, TurboQuantLinear):
            replacements.append((name, "linear", module))
        elif isinstance(module, TurboQuantEmbedding):
            replacements.append((name, "embedding", module))

    print(f"Decompressing {len(replacements)} layers to bf16...")
    for name, ltype, module in replacements:
        if ltype == "linear":
            W = module.dequantize()  # (out, in) bf16
            has_bias = module.bias is not None
            linear = nn.Linear(
                module.in_features, module.out_features,
                bias=has_bias, dtype=W.dtype, device=W.device,
            )
            linear.weight.data.copy_(W)
            if has_bias:
                linear.bias.data.copy_(module.bias)
            _set_module(model, name, linear)
        elif ltype == "embedding":
            # Dequantize all embedding rows
            all_ids = torch.arange(module.num_embeddings, device=module.indices_packed.device)
            W = module(all_ids)  # (vocab, dim)
            emb = nn.Embedding(
                module.num_embeddings, module.embedding_dim,
                padding_idx=module.padding_idx,
                dtype=W.dtype, device=W.device,
            )
            emb.weight.data.copy_(W)
            _set_module(model, name, emb)

        # Free the packed data
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Re-tie weights if needed
    if (hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False)
            and hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens")):
        model.lm_head.weight = model.model.embed_tokens.weight

    print("Decompression complete.")
    return model
