# Train on TPU, run everywhere 🚀

Convert JAX LoRA leaves saved as `.npz` files (from Tunix/JAX) into HuggingFace PEFT LoRA adapters in safetensors format.

This tool enables you to train LoRA models using JAX/Flax on **Google's TPU** (via platforms like Kaggle), then convert them to the widely-supported PEFT format (using safetensors) for inference and evaluation with PyTorch and HuggingFace Transformers on **GPU**.

## Overview

This project provides:
- **`jax_to_safetensors.py`**: Converts JAX `.npz` LoRA weights to PEFT adapter format (safetensors)
- **`eval_safetensors_gsm8k.py`**: Evaluates models on the GSM8K math reasoning benchmark
- **`chat.py`**: Interactive chat interface to compare base and LoRA models side-by-side

## Installation

```bash
pip install torch transformers datasets peft safetensors numpy
```

## Quick Start

### Step 1: Train a LoRA Model with JAX on TPU

Train `gemma-3-1b-it` using the [GRPO Starter notebook on Kaggle](https://www.kaggle.com/code/windmaple/grpo-demo-gemma3-1b) with **Google's TPU** for accelerated training.

Add this code block at the end of the notebook to export the trained LoRA weights:

```python
# Save final LoRA leaves
import numpy as np
from jax import tree_util

trained_lora_state = nnx.state(lora_policy, nnx.LoRAParam)
flat_trained = tree_util.tree_flatten_with_path(trained_lora_state)[0]

trained_dict = {
    "/".join(str(k) for k in path): np.array(leaf)
    for path, leaf in flat_trained
}

out_path = "/kaggle/working/trained_lora_leaves_final.npz"
np.savez(out_path, **trained_dict)
print(f"Saved trained LoRA leaves to: {out_path}")
print("Num leaves:", len(trained_dict))
```

### Step 2: Convert to PEFT Adapter (Safetensors)

Download `trained_lora_leaves_final.npz` and convert it to a PEFT adapter in safetensors format:

```bash
python3 jax_to_safetensors.py --npz trained_lora_leaves_final.npz --out_dir peft_adapter
```

**Output:**
```
Wrote PEFT adapter to peft_adapter
Num tensors: 364
```

The `peft_adapter` folder will contain two files:
- `adapter_config.json` - PEFT configuration metadata
- `adapter_model.safetensors` - LoRA weights in safetensors format

### Step 3: Evaluate on GSM8K

Evaluate the model's math reasoning ability on the first 400 questions from GSM8K:

#### With LoRA Adapter (Fine-tuned)

```bash
python3 eval_safetensors_gsm8k.py --base_id google/gemma-3-1b-it \
    --batch_size=128 \
    --adapter_dir=peft_adapter \
    --save_outputs=debug.json
```

**Results:**
```
============================================================
EVALUATION RESULTS
============================================================
Accuracy: 44.25% (177/400), Partial: 45.50%, Formatted: 91.50%
```

#### Without LoRA Adapter (Base Model)

```bash
python3 eval_safetensors_gsm8k.py --base_id google/gemma-3-1b-it \
    --batch_size=128 \
    --save_outputs=debug.json
```

**Results:**
```
============================================================
EVALUATION RESULTS
============================================================
Accuracy: 13.75% (55/400), Partial: 16.50%, Formatted: 48.75%
```

### 📈 Performance Improvement

The LoRA-adapted model shows a **significant accuracy jump from 13.75% to 44.25%** on GSM8K, demonstrating the effectiveness of the fine-tuning approach.

## Interactive Chat Interface

Compare the base model and LoRA model side-by-side:

```bash
python3 chat.py --adapter_dir=peft_adapter
```

**Example:**

```
QUESTION> Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. She 
sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
How much in dollars does she make every day at the farmers' market?

────────────────────────────────────────────────────────────────────────────────
  BASE MODEL
────────────────────────────────────────────────────────────────────────────────
Okay, let's break this problem down step-by-step.

<reasoning>The problem states that Janet lays 16 eggs per day, she eats 3 for 
breakfast, and bakes muffins with 4. The remaining eggs are sold for $2 each. 
We need to find the total earnings. The number of eggs remaining after breakfast 
and muffins is 16 - 3 - 4 = 9 eggs. Therefore, she makes $2 per egg, and she 
sells 9 eggs, so her total earnings are $2 * 9 = $18. </answer>

────────────────────────────────────────────────────────────────────────────────
  LoRA MODEL
────────────────────────────────────────────────────────────────────────────────
Okay, let's break this problem down step-by-step.

<reasoning>
The problem states that Janet's ducks lay 16 eggs per day, and she eats 3 for 
breakfast and bakes muffins for friends with 4. This means she has 16 - 3 - 4 = 
9 eggs remaining. She sells each duck egg for $2, so she makes 9 * $2 = $18 per 
day.
</reasoning>

<answer>18</answer>
```

Notice how the LoRA model properly formats its answer with the `<answer>` tags, showing improved instruction-following capability.

## Command-Line Options

### `jax_to_safetensors.py`

```bash
python3 jax_to_safetensors.py --npz <path_to_npz> --out_dir <output_directory>
```

### `eval_safetensors_gsm8k.py`

```bash
python3 eval_safetensors_gsm8k.py \
    --base_id <model_id> \
    --adapter_dir <path_to_adapter> \
    --num_samples 400 \
    --batch_size 64 \
    --max_new_tokens 768 \
    --dtype bf16 \
    --save_outputs <output.json>
```

### `chat.py`

```bash
python3 chat.py \
    --base_id <model_id> \
    --adapter_dir <path_to_adapter> \
    --dtype bf16
```

## Evaluation Metrics

- **Accuracy**: Percentage of problems where the model provides the correct answer in the expected `<answer>` tag format
- **Partial**: Percentage where the correct answer appears anywhere in the output
- **Formatted**: Percentage of responses that properly use the `<answer>` tag format

## Use Cases

- Converting JAX-trained LoRA models to PyTorch for production deployment
- Training on **Google's TPU** (Kaggle) and deploying on **GPU** infrastructure for inference
- Benchmarking fine-tuned models on math reasoning tasks
- Comparing base model vs. fine-tuned model performance

## License

MIT

## Acknowledgments

- [GRPO Demo on Kaggle](https://www.kaggle.com/code/windmaple/grpo-demo-gemma3-1b) for the training approach
- [GSM8K dataset](https://huggingface.co/datasets/gsm8k) for evaluation
- HuggingFace PEFT library for LoRA adapter support

