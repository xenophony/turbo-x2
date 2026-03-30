#!/usr/bin/env python3
"""CLI: Compare quality of original vs TurboQuant-quantized model.

Usage:
    python eval.py --original Qwen/Qwen2.5-0.5B-Instruct --quantized ./quantized/qwen-0.5b-2bit
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.model import load_quantized_model


EVAL_PROMPTS = [
    "The capital of France is",
    "In 1969, the first person to walk on the moon was",
    "The largest planet in our solar system is",
    "Water boils at",
    "The theory of relativity was developed by",
]

PERPLEXITY_TEXT = (
    "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey "
    "building, and the tallest structure in Paris. Its base is square, measuring 125 "
    "metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed "
    "the Washington Monument to become the tallest man-made structure in the world, a "
    "title it held for 41 years until the Chrysler Building in New York City was "
    "finished in 1930. It was the first structure in the world to surpass both the "
    "200-metre and 300-metre mark in height. Due to the addition of a broadcasting "
    "aerial at the top of the tower in 1957, it is now taller than the Chrysler "
    "Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the "
    "second tallest free-standing structure in France after the Millau Viaduct."
)


def compute_perplexity(model, tokenizer, text: str, device: str) -> float:
    """Compute perplexity of a model on a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()


def generate_comparison(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 50) -> str:
    """Generate text from a model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def get_memory_mb(model) -> float:
    """Estimate model memory in MB."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / 1024**2


def main():
    parser = argparse.ArgumentParser(description="Evaluate TurboQuant quantization quality")
    parser.add_argument("--original", type=str, required=True, help="Original HuggingFace model ID")
    parser.add_argument("--quantized", type=str, required=True, help="Path to quantized model directory")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load original model
    print(f"Loading original model: {args.original}...")
    tokenizer = AutoTokenizer.from_pretrained(args.original)
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original, torch_dtype=torch.bfloat16, device_map=device,
    )
    original_model.eval()

    # Load quantized model
    print(f"Loading quantized model: {args.quantized}...")
    quant_model = load_quantized_model(args.quantized, device=device)
    quant_model.eval()

    # Memory comparison
    orig_mem = get_memory_mb(original_model)
    quant_mem = get_memory_mb(quant_model)
    print(f"\n{'='*60}")
    print(f"Memory Usage:")
    print(f"  Original:  {orig_mem:.1f} MB")
    print(f"  Quantized: {quant_mem:.1f} MB")
    print(f"  Ratio:     {orig_mem / quant_mem:.1f}x compression")

    # Perplexity comparison
    print(f"\n{'='*60}")
    print(f"Perplexity (lower is better):")
    orig_ppl = compute_perplexity(original_model, tokenizer, PERPLEXITY_TEXT, device)
    quant_ppl = compute_perplexity(quant_model, tokenizer, PERPLEXITY_TEXT, device)
    ppl_change = (quant_ppl - orig_ppl) / orig_ppl * 100
    print(f"  Original:  {orig_ppl:.2f}")
    print(f"  Quantized: {quant_ppl:.2f}")
    print(f"  Change:    {ppl_change:+.1f}%")

    # Generation comparison
    print(f"\n{'='*60}")
    print(f"Generation Comparison:")
    for prompt in EVAL_PROMPTS:
        orig_text = generate_comparison(original_model, tokenizer, prompt, device)
        quant_text = generate_comparison(quant_model, tokenizer, prompt, device)
        print(f"\n  Prompt: {prompt}")
        print(f"  Original:  {orig_text}")
        print(f"  Quantized: {quant_text}")

    print(f"\n{'='*60}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
