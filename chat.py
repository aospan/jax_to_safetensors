#!/usr/bin/env python3
"""
Interactive chat to compare base model vs LoRA fine-tuned model outputs.

Usage:
  # Compare base vs LoRA:
  python chat.py --base_id google/gemma-3-1b-it --adapter_dir ./peft_adapter
  
  # Use base model only:
  python chat.py --base_id google/gemma-3-1b-it
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


SYSTEM_PROMPT = """You are given a problem. Think about the problem and \
provide your reasoning. Place it between <reasoning> and </reasoning>. \
Then, provide the final answer (i.e., just one numerical value) between \
<answer> and </answer>."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def clean_output(full_output: str) -> str:
    """Extract model response, removing prompt echo."""
    marker = "<start_of_turn>model"
    if marker in full_output:
        response = full_output.split(marker)[-1]
        response = response.replace("<end_of_turn>", "").strip()
        return response
    return full_output.strip()


def generate(model, tokenizer, prompt: str, **gen_kwargs) -> str:
    """Generate text from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Compare base vs LoRA model outputs")
    parser.add_argument("--base_id", default="google/gemma-3-1b-it", help="HuggingFace model ID")
    parser.add_argument("--adapter_dir", default=None, help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="Model dtype")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_id, torch_dtype=dtype_map[args.dtype], device_map="auto"
    ).eval()
    
    tuned_model = None
    if args.adapter_dir:
        tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir).eval()

    gen_kwargs = {
        "max_new_tokens": 768,
        "do_sample": True,
        "temperature": 1e-4,
        "top_k": 1,
        "top_p": 1.0,
        "eos_token_id": [1, 106],
        "use_cache": True,
    }

    if tuned_model:
        print_header("Interactive Base vs LoRA Comparison")
        print(f"  Base model: {args.base_id}")
        print(f"  Adapter: {args.adapter_dir}")
    else:
        print_header("Interactive Chat (Base Model Only)")
        print(f"  Model: {args.base_id}")
    print(f"  Settings: temp=1e-4, top_k=1, max_tokens=768")
    print(f"\n  Type 'quit' or empty line to exit.\n")

    while True:
        question = input("QUESTION> ").strip()
        if not question or question.lower() in ["quit", "exit", "q"]:
            print("\nExiting.")
            break

        prompt = TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)

        if tuned_model:
            # Base model output
            print_header("BASE MODEL")
            with tuned_model.disable_adapter():
                base_output = generate(tuned_model, tokenizer, prompt, **gen_kwargs)
            print(clean_output(base_output))

            # LoRA model output
            print_header("LoRA MODEL")
            tuned_output = generate(tuned_model, tokenizer, prompt, **gen_kwargs)
            print(clean_output(tuned_output))
        else:
            # Base model only
            print_header("MODEL RESPONSE")
            base_output = generate(base_model, tokenizer, prompt, **gen_kwargs)
            print(clean_output(base_output))
        
        print()


if __name__ == "__main__":
    main()

