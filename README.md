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

### Performance Improvement

The LoRA-adapted model shows a **significant accuracy jump from 13.75% to 44.25%** on GSM8K.

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

## Evaluation Metrics

- **Accuracy**: Percentage of problems where the model provides the correct answer in the expected `<answer>` tag format
- **Partial**: Percentage where the correct answer appears anywhere in the output
- **Formatted**: Percentage of responses that properly use the `<answer>` tag format

## Dive Deep: Understanding the File Formats

This section explains the binary file formats involved in the conversion process.

### NPZ Format (Input)

NPZ is a zipped archive containing multiple individual `.npy` files, each representing one NumPy array.

Let's extract the NPZ file obtained from training:

```bash
unzip -l trained_lora_leaves_final.npz
```

```
Archive:  trained_lora_leaves_final.npz
 extracting: layers/0/attn/attn_vec_einsum/w_lora_a/value.npy
 extracting: layers/0/attn/attn_vec_einsum/w_lora_b/value.npy
 extracting: layers/0/attn/kv_einsum/w_lora_a/value.npy
 extracting: layers/0/attn/kv_einsum/w_lora_b/value.npy
 extracting: layers/0/attn/q_einsum/w_lora_a/value.npy
 extracting: layers/0/attn/q_einsum/w_lora_b/value.npy
...
```

Let's explore one `.npy` file with a hex dump:

```bash
xxd -l 128 layers/0/attn/attn_vec_einsum/w_lora_a/value.npy
```

```
00000000: 934e 554d 5059 0100 7600 7b27 6465 7363  .NUMPY..v.{'desc
00000010: 7227 3a20 273c 5632 272c 2027 666f 7274  r': '<V2', 'fort
00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
00000030: 652c 2027 7368 6170 6527 3a20 2834 2c20  e, 'shape': (4, 
00000040: 3235 362c 2036 3429 2c20 7d20 2020 2020  256, 64), }     
00000050: 2020 2020 2020 2020 2020 2020 2020 2020                  
00000060: 2020 2020 2020 2020 2020 2020 2020 2020                  
00000070: 2020 2020 2020 2020 2020 2020 2020 200a                 .
```

According to the [NumPy Format Specification](https://numpy.org/neps/nep-0001-npy-format.html), the file starts with:
- Magic string: `\x93NUMPY`
- Version bytes
- Header describing the array (a Python dictionary with `dtype`, `fortran_order`, and `shape`)
- The most important field is `shape`: `(4, 256, 64)` in this example

The header is followed by the raw binary array data.

### Safetensors Format (Output)

Let's examine the converted safetensors file:

```bash
xxd -l 128 peft_adapter/adapter_model.safetensors
```

```
00000000: d8bc 0000 0000 0000 7b22 6261 7365 5f6d  ........{"base_m
00000010: 6f64 656c 2e6d 6f64 656c 2e6d 6f64 656c  odel.model.model
00000020: 2e6c 6179 6572 732e 302e 6d6c 702e 646f  .layers.0.mlp.do
00000030: 776e 5f70 726f 6a2e 6c6f 7261 5f41 2e77  wn_proj.lora_A.w
00000040: 6569 6768 7422 3a7b 2264 7479 7065 223a  eight":{"dtype":
00000050: 2246 3136 222c 2273 6861 7065 223a 5b36  "F16","shape":[6
00000060: 342c 3639 3132 5d2c 2264 6174 615f 6f66  4,6912],"data_of
00000070: 6673 6574 7322 3a5b 302c 3838 3437 3336  fsets":[0,884736
```

According to the [safetensors format specification](https://huggingface.co/docs/safetensors/en/index), the format is: **header JSON + binary blobs**.

From the hex dump above:
- `dtype: "F16"` → float16
- `shape: [64, 6912]` → tensor dimensions
- `data_offsets: [0, 884736]` → byte range where this tensor's data is stored

The tensor name `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight` tells us:
- `base_model.model.model` → PEFT wraps a base HuggingFace model as `self.base_model`
- `.layers.0.mlp.down_proj` → target module in the base model
- `.lora_A.weight` → LoRA A matrix for that module

### PEFT Adapter Configuration

The `adapter_config.json` file tells PEFT how and where to plug the LoRA weights into the base model:

```bash
cat peft_adapter/adapter_config.json
```

```json
{
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.0,
  "bias": "none",
  "target_modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ]
}
```

Key parameters:
- `r: 64` → LoRA rank (dimensionality of the low-rank adaptation)
- `lora_alpha: 64` → LoRA scaling factor
- `target_modules` → Which layers in the base model receive LoRA adapters

## Use Cases

- Converting JAX-trained LoRA models to PyTorch for production deployment
- Training on **Google's TPU** and deploying on **GPU** infrastructure for inference
- Benchmarking fine-tuned models on math reasoning tasks
- Comparing base model vs. fine-tuned model performance

## License

MIT

## Acknowledgments

- [GRPO Demo on Kaggle](https://www.kaggle.com/code/windmaple/grpo-demo-gemma3-1b) for the training approach
- [GSM8K dataset](https://huggingface.co/datasets/gsm8k) for evaluation
- HuggingFace PEFT library for LoRA adapter support

