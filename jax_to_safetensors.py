#!/usr/bin/env python3
"""
jax_to_safetensors.py

Convert JAX LoRA leaves saved as an .npz (from Tunix/JAX) into a HuggingFace PEFT LoRA adapter.

Input NPZ is expected to contain ONLY LoRA leaves keyed in one of two formats:

Format 1 (plain paths):
  layers/0/attn/q_einsum/w_lora_a/value
  layers/0/attn/q_einsum/w_lora_b/value
  layers/0/attn/kv_einsum/w_lora_a/value
  layers/0/attn/kv_einsum/w_lora_b/value
  layers/0/attn/attn_vec_einsum/w_lora_a/value
  layers/0/attn/attn_vec_einsum/w_lora_b/value
  layers/0/mlp/gate_proj/kernel_lora_a/value
  layers/0/mlp/gate_proj/kernel_lora_b/value
  layers/0/mlp/up_proj/kernel_lora_a/value
  layers/0/mlp/up_proj/kernel_lora_b/value
  layers/0/mlp/down_proj/kernel_lora_a/value
  layers/0/mlp/down_proj/kernel_lora_b/value
  ... for all layers

Format 2 (bracketed paths):
  ['layers']/[0]/['attn']/['q_einsum']/['w_lora_a']/.value.npy
  ['layers']/[0]/['attn']/['q_einsum']/['w_lora_b']/.value.npy
  ['layers']/[0]/['attn']/['kv_einsum']/['w_lora_a']/.value.npy
  ['layers']/[0]/['attn']/['kv_einsum']/['w_lora_b']/.value.npy
  ['layers']/[0]/['attn']/['attn_vec_einsum']/['w_lora_a']/.value.npy
  ['layers']/[0]/['attn']/['attn_vec_einsum']/['w_lora_b']/.value.npy
  ['layers']/[0]/['mlp']/['gate_proj']/['kernel_lora_a']/.value.npy
  ['layers']/[0]/['mlp']/['gate_proj']/['kernel_lora_b']/.value.npy
  ... for all layers

Gemma 3 1B mapping rules:
- q_einsum: A -> q_proj.lora_A, B -> q_proj.lora_B (flatten heads)
- kv_einsum: A shared for k/v, B split into k/v halves
- attn_vec_einsum: A flatten heads -> o_proj.lora_A, B -> o_proj.lora_B
- mlp gate/up/down -> corresponding proj lora_A/B
- Store keys WITHOUT ".default" segment (PEFT inserts adapter name automatically)

Usage:
  python3 jax_to_safetensors.py \
    --npz trained_lora_leaves.npz \
    --out_dir ./peft_adapter \
    --rank 64 --alpha 64 --out_dtype fp16
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from safetensors.torch import save_file


PEFT_PREFIX = "base_model.model.model"


@dataclass
class ConversionConfig:
    """Configuration for JAX to PEFT conversion."""
    npz_path: str
    out_dir: str
    rank: int
    alpha: int
    out_dtype: str
    head_dim: int


def peft_key(rel_path: str) -> str:
    """Generate PEFT-formatted key from relative path."""
    return f"{PEFT_PREFIX}.{rel_path}"


def bf16_blob_to_fp32(x: np.ndarray) -> np.ndarray:
    """Convert numpy 'void' bf16-bit blobs into float32."""
    u16 = x.view(np.uint16)
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)


def normalize_array(x: np.ndarray) -> np.ndarray:
    """Ensure we end up with a real numeric ndarray (float32)."""
    x = np.asarray(x)
    if x.dtype.kind == "V":  # numpy.void -> likely bf16
        return bf16_blob_to_fp32(x)
    return x.astype(np.float32)


def array_to_tensor(arr: np.ndarray, dtype: str) -> torch.Tensor:
    """Convert numpy array to torch tensor with specified dtype."""
    arr = normalize_array(arr)
    t = torch.from_numpy(arr)
    
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    
    return t.to(dtype_map.get(dtype, torch.float32)).contiguous().clone()


def load_npz_leaves(npz_path: str) -> Dict[str, np.ndarray]:
    """Load and normalize NPZ file contents."""
    npz = np.load(npz_path, allow_pickle=True)
    
    leaves = {}
    for k in npz.files:
        v = npz[k]
        # Handle 0-d object arrays
        if isinstance(v, np.ndarray) and v.shape == () and v.dtype.kind == "O":
            v = v.item()
        leaves[k] = v
    
    return leaves


def normalize_npz_key(key: str) -> str:
    """
    Normalize NPZ keys from different formats to standard format.
    
    Handles both:
    - Original: layers/0/attn/q_einsum/w_lora_a/value
    - Bracketed: ['layers']/[0]/['attn']/['q_einsum']/['w_lora_a']/.value.npy
    
    Returns normalized format: layers/0/attn/q_einsum/w_lora_a/value
    """
    # Convert ['key'] -> key and [number] -> number
    # Pattern matches ['...'] or [number]
    normalized = re.sub(r"\['([^']+)'\]", r"\1", key)  # ['key'] -> key
    normalized = re.sub(r"\[(\d+)\]", r"\1", normalized)  # [number] -> number
    
    # Remove leading slash if present after bracket removal
    if normalized.startswith("/"):
        normalized = normalized[1:]
    
    # Handle /.value.npy suffix (convert to /value)
    if normalized.endswith("/.value.npy"):
        normalized = normalized[:-10] + "/value"
    # Handle /.value suffix (convert to /value)
    elif normalized.endswith("/.value"):
        normalized = normalized[:-7] + "/value"
    # Handle .value.npy suffix (convert to /value)
    elif normalized.endswith(".value.npy"):
        normalized = normalized[:-10] + "/value"
    
    return normalized


def strip_value_suffix(path: str) -> str:
    """Remove /value suffix from path if present."""
    return path[:-6] if path.endswith("/value") else path


def process_mlp_layer(state_dict: Dict[str, torch.Tensor], layer_idx: int, 
                     module: str, matrix_type: str, weight: np.ndarray, 
                     out_dtype: str) -> None:
    """Process MLP layer LoRA weights."""
    lora_matrix = "lora_A" if matrix_type == "a" else "lora_B"
    key = peft_key(f"layers.{layer_idx}.mlp.{module}.{lora_matrix}.weight")
    state_dict[key] = array_to_tensor(weight.T, out_dtype)


def process_q_einsum(state_dict: Dict[str, torch.Tensor], layer_idx: int, 
                    matrix_type: str, weight: np.ndarray, rank: int, 
                    out_dtype: str) -> None:
    """Process query projection LoRA weights."""
    if matrix_type == "a":
        key = peft_key(f"layers.{layer_idx}.self_attn.q_proj.lora_A.weight")
        state_dict[key] = array_to_tensor(weight.T, out_dtype)
    else:
        # Flatten (r, heads, head_dim) to (r, hidden_attn)
        weight_flat = weight.reshape(rank, -1)
        key = peft_key(f"layers.{layer_idx}.self_attn.q_proj.lora_B.weight")
        state_dict[key] = array_to_tensor(weight_flat.T, out_dtype)


def process_kv_einsum(state_dict: Dict[str, torch.Tensor], layer_idx: int,
                     matrix_type: str, weight: np.ndarray, rank: int,
                     head_dim: int, out_dtype: str) -> None:
    """Process key/value projection LoRA weights."""
    if matrix_type == "a":
        # Shared A for both k and v - need separate copies for safetensors
        weight_t = weight.T
        key_k = peft_key(f"layers.{layer_idx}.self_attn.k_proj.lora_A.weight")
        key_v = peft_key(f"layers.{layer_idx}.self_attn.v_proj.lora_A.weight")
        state_dict[key_k] = array_to_tensor(weight_t, out_dtype)
        state_dict[key_v] = array_to_tensor(weight_t, out_dtype)
    else:
        # Split B into k and v components
        weight_flat = weight.reshape(rank, -1)
        weight_k = weight_flat[:, :head_dim]
        weight_v = weight_flat[:, head_dim:]
        
        key_k = peft_key(f"layers.{layer_idx}.self_attn.k_proj.lora_B.weight")
        key_v = peft_key(f"layers.{layer_idx}.self_attn.v_proj.lora_B.weight")
        state_dict[key_k] = array_to_tensor(weight_k.T, out_dtype)
        state_dict[key_v] = array_to_tensor(weight_v.T, out_dtype)


def process_attn_vec_einsum(state_dict: Dict[str, torch.Tensor], layer_idx: int,
                           matrix_type: str, weight: np.ndarray, rank: int,
                           out_dtype: str) -> None:
    """Process attention output projection LoRA weights."""
    if matrix_type == "a":
        # Flatten (heads, head_dim, r) to (hidden_attn, r)
        weight_flat = weight.reshape(-1, rank)
        key = peft_key(f"layers.{layer_idx}.self_attn.o_proj.lora_A.weight")
        state_dict[key] = array_to_tensor(weight_flat.T, out_dtype)
    else:
        key = peft_key(f"layers.{layer_idx}.self_attn.o_proj.lora_B.weight")
        state_dict[key] = array_to_tensor(weight.T, out_dtype)


def convert_jax_to_peft(leaves: Dict[str, np.ndarray], config: ConversionConfig) -> Dict[str, torch.Tensor]:
    """Convert JAX LoRA leaves to PEFT state dict."""
    state_dict = {}
    
    attn_pattern = re.compile(
        r"layers/(\d+)/attn/(q_einsum|kv_einsum|attn_vec_einsum)/w_lora_(a|b)$"
    )
    mlp_pattern = re.compile(
        r"layers/(\d+)/mlp/(gate_proj|up_proj|down_proj)/kernel_lora_(a|b)$"
    )
    
    for path, weight in leaves.items():
        # Normalize key format (handles both bracketed and plain formats)
        path = normalize_npz_key(path)
        path = strip_value_suffix(path)
        
        # Try MLP pattern
        if match := mlp_pattern.match(path):
            layer_idx = int(match.group(1))
            module = match.group(2)
            matrix_type = match.group(3)
            process_mlp_layer(state_dict, layer_idx, module, matrix_type, weight, config.out_dtype)
            continue
        
        # Try attention pattern
        if match := attn_pattern.match(path):
            layer_idx = int(match.group(1))
            module = match.group(2)
            matrix_type = match.group(3)
            
            if module == "q_einsum":
                process_q_einsum(state_dict, layer_idx, matrix_type, weight, config.rank, config.out_dtype)
            elif module == "kv_einsum":
                process_kv_einsum(state_dict, layer_idx, matrix_type, weight, config.rank, config.head_dim, config.out_dtype)
            elif module == "attn_vec_einsum":
                process_attn_vec_einsum(state_dict, layer_idx, matrix_type, weight, config.rank, config.out_dtype)
    
    return state_dict


def create_adapter_config(rank: int, alpha: int) -> Dict[str, Any]:
    """Create PEFT adapter configuration."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    }


def save_adapter(state_dict: Dict[str, torch.Tensor], config: ConversionConfig) -> None:
    """Save adapter model and configuration."""
    out_path = Path(config.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save tensors
    save_file(state_dict, str(out_path / "adapter_model.safetensors"))
    
    # Save config
    adapter_config = create_adapter_config(config.rank, config.alpha)
    with open(out_path / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    
    print(f"Wrote PEFT adapter to {config.out_dir}")
    print(f"Num tensors: {len(state_dict)}")


def parse_args() -> ConversionConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert JAX LoRA to PEFT format")
    parser.add_argument("--npz", required=True, help="Path to trained_lora_leaves.npz")
    parser.add_argument("--out_dir", default="./peft_adapter", help="Output directory")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--out_dtype", choices=["fp32", "fp16", "bf16"], default="fp16", help="Output dtype")
    parser.add_argument("--head_dim", type=int, default=256, help="Attention head dimension")
    
    args = parser.parse_args()
    
    return ConversionConfig(
        npz_path=args.npz,
        out_dir=args.out_dir,
        rank=args.rank,
        alpha=args.alpha,
        out_dtype=args.out_dtype,
        head_dim=args.head_dim,
    )


def main() -> None:
    """Main conversion workflow."""
    config = parse_args()
    leaves = load_npz_leaves(config.npz_path)
    state_dict = convert_jax_to_peft(leaves, config)
    save_adapter(state_dict, config)


if __name__ == "__main__":
    main()

