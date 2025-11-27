#!/usr/bin/env python3
"""
Evaluate language models on the GSM8K math reasoning benchmark.

This script evaluates a model's ability to solve grade school math problems
by generating solutions with reasoning and extracting numerical answers.

Usage:
    # Basic evaluation
    python eval_safetensors_gsm8k.py --base_id google/gemma-3-1b-it

    # With adapter and custom settings
    python eval_safetensors_gsm8k.py --base_id google/gemma-3-1b-it \
        --adapter_dir ./peft_adapter

    # Save outputs for analysis
    python eval_safetensors_gsm8k.py --base_id google/gemma-3-1b-it \
        --save_outputs results.json
"""
import argparse
import json
import re
import time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

# Prompt template
SYSTEM_PROMPT = """You are given a problem. Think about the problem and \
provide your reasoning. Place it between <reasoning> and </reasoning>. \
Then, provide the final answer (i.e., just one numerical value) between \
<answer> and </answer>."""


TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

# Regex patterns
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
ANSWER_TAG_RE = re.compile(r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>", re.DOTALL)

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_prompt(question: str) -> str:
    return TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)


def normalize_num(x: str) -> str:
    """Normalize a numeric string for comparison."""
    try:
        if "." in x:
            f = float(x)
            if abs(f - round(f)) < 1e-9:
                return str(int(round(f)))
            return str(f).rstrip("0").rstrip(".")
        return str(int(x))
    except (ValueError, TypeError):
        return x.strip()


def extract_gold(answer_text: str) -> str | None:
    """Extract gold answer from GSM8K format."""
    if "####" in answer_text:
        tail = answer_text.split("####")[-1].strip()
        m = NUM_RE.search(tail)
        if m:
            return normalize_num(m.group(0))
    nums = NUM_RE.findall(answer_text)
    return normalize_num(nums[-1]) if nums else None


def extract_prediction(model_text: str) -> tuple[str | None, str | None, bool]:
    """Extract prediction from model output.
    
    Returns:
        (formatted_answer, any_answer, has_format)
    """
    # Try to extract from <answer> tags
    m = ANSWER_TAG_RE.search(model_text)
    formatted = normalize_num(m.group(1)) if m else None
    has_format = m is not None
    
    # Extract any number as fallback
    nums = NUM_RE.findall(model_text)
    any_answer = normalize_num(nums[-1]) if nums else None
    
    return formatted, any_answer, has_format

def load_model(base_id: str, adapter_dir: str | None, torch_dtype):
    """Load base model and optional adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    
    model = AutoModelForCausalLM.from_pretrained(base_id, **model_kwargs).eval()
    
    if adapter_dir:
        if PeftModel is None:
            raise RuntimeError("peft not installed but --adapter_dir given")
        model = PeftModel.from_pretrained(model, adapter_dir).eval()
    
    return tokenizer, model


def print_metrics(correct: int, partial: int, formatted: int, total: int, end="\n"):
    """Print evaluation metrics."""
    acc = correct / total * 100
    partial_acc = partial / total * 100
    format_acc = formatted / total * 100
    print(f"Accuracy: {acc:.2f}% ({correct}/{total}), "
          f"Partial: {partial_acc:.2f}%, Formatted: {format_acc:.2f}%", end=end)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--base_id", default="google/gemma-3-1b-it")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=400, help="Total samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--save_outputs", type=str, default=None, help="Save raw inputs/outputs to JSON file")
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Model: {args.base_id}")
    print(f"  Adapter: {args.adapter_dir or 'None'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Dtype: {args.dtype}")
    if args.save_outputs:
        print(f"  Save outputs: {args.save_outputs}")

    # Load model
    tokenizer, model = load_model(
        args.base_id, 
        args.adapter_dir, 
        DTYPE_MAP[args.dtype]
    )

    # Load dataset
    dataset = load_dataset("gsm8k", "main", split="test").select(range(args.num_samples))

    # Generation config (greedy sampling)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": 1e-4,
        "top_k": 1,
        "eos_token_id": [1, 106],
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    print(f"\n{'='*60}")
    print(f"STARTING EVALUATION")
    print(f"{'='*60}")
    print(f"Samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches: {(args.num_samples + args.batch_size - 1) // args.batch_size}\n")

    # Evaluate with batching
    print(f"Running inference...")
    start_time = time.time()
    correct = partial = formatted = 0
    total_processed = 0
    
    # Store outputs if requested
    saved_outputs = [] if args.save_outputs else None
    
    # Process in batches
    for batch_start in range(0, args.num_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_size = batch_end - batch_start
        
        # Get batch data (dataset slicing returns dict of lists)
        batch_data = dataset[batch_start:batch_end]
        gold_answers = [extract_gold(ans) for ans in batch_data["answer"]]
        prompts = [build_prompt(q) for q in batch_data["question"]]
        
        # Tokenize batch
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        )
        
        # Move to device and generate
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode and evaluate each example in batch
        for i, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=False)
            pred_fmt, pred_any, has_format = extract_prediction(text)
            gold = gold_answers[i]
            
            # Save raw input/output if requested
            if saved_outputs is not None:
                saved_outputs.append({
                    "index": batch_start + i,
                    "question": batch_data["question"][i],
                    "gold_answer": batch_data["answer"][i],
                    "gold_extracted": gold,
                    "prompt": prompts[i],
                    "output": text,
                    "predicted_formatted": pred_fmt,
                    "predicted_any": pred_any,
                    "has_format": has_format,
                    "correct": pred_fmt == gold and gold is not None,
                })

            # Count formatted answers
            if has_format and pred_fmt is not None:
                formatted += 1

            # Count correct answers
            if pred_fmt == gold and gold is not None:
                correct += 1
                partial += 1
            elif pred_any == gold and gold is not None:
                partial += 1

        total_processed = batch_end
        
        # Progress update
        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        print(f"    [{total_processed}/{args.num_samples}] ", end="")
        print_metrics(correct, partial, formatted, total_processed, end="")
        print(f" | {rate:.2f} samples/sec")

    # Final results
    elapsed_time = time.time() - start_time
    samples_per_sec = total_processed / elapsed_time
    
    print(f"\nInference complete!\n")
    
    # Save outputs to JSON if requested
    if saved_outputs is not None:
        print(f"Saving outputs to {args.save_outputs}...")
        output_data = {
            "metadata": {
                "base_id": args.base_id,
                "adapter_dir": args.adapter_dir,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "dtype": args.dtype,
                "elapsed_time": elapsed_time,
                "throughput": samples_per_sec,
            },
            "results": {
                "correct": correct,
                "partial": partial,
                "formatted": formatted,
                "total": total_processed,
                "accuracy": correct / total_processed * 100,
                "partial_accuracy": partial / total_processed * 100,
                "format_accuracy": formatted / total_processed * 100,
            },
            "outputs": saved_outputs
        }
        with open(args.save_outputs, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(saved_outputs)} outputs to {args.save_outputs}\n")
    
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print_metrics(correct, partial, formatted, total_processed)
    print(f"\nTiming:")
    print(f"  Total time: {elapsed_time:.1f}s")
    print(f"  Throughput: {samples_per_sec:.2f} samples/sec")
    print(f"  Avg per sample: {elapsed_time/total_processed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

