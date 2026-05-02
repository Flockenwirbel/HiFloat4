#!/usr/bin/env python3
"""Verify transformer model loading correctness."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from hifloat4.wan_model_utils import load_wan_model

def verify_model():
    print("Verifying model loading...")
    
    # Load model
    model = load_wan_model('/home/dataset/Wan2.2-I2V-A14B', device='cpu', dtype=torch.bfloat16)
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"Config:")
    print(f"  in_channels: {model.config.in_channels}")
    print(f"  out_channels: {model.config.out_channels}")
    print(f"  patch_size: {model.config.patch_size}")
    print(f"  image_dim: {model.config.image_dim}")
    print(f"  num_layers: {model.config.num_layers}")
    
    # Check patch_embedding
    print(f"\npatch_embedding:")
    print(f"  weight shape: {model.patch_embedding.weight.shape}")
    print(f"  bias shape: {model.patch_embedding.bias.shape}")
    
    # Check if img_emb exists
    has_img_emb = hasattr(model, 'img_emb') and model.img_emb is not None
    print(f"\nHas img_emb: {has_img_emb}")
    
    if has_img_emb:
        print(f"  img_emb type: {type(model.img_emb)}")
    else:
        print("  WARNING: No img_emb module (this is expected for I2V without CLIP image conditioning)")
    
    # Check a few blocks
    print(f"\nFirst block:")
    block0 = model.blocks[0]
    print(f"  type: {type(block0).__name__}")
    if hasattr(block0, 'cross_attn'):
        print(f"  has cross_attn: True")
        if hasattr(block0.cross_attn, 'to_k'):
            print(f"  cross_attn.to_k weight shape: {block0.cross_attn.to_k.weight.shape}")
    else:
        print(f"  has cross_attn: False")
    
    # Check if weights look reasonable (not all zeros or NaN)
    print(f"\nWeight statistics (first layer):")
    w = model.patch_embedding.weight
    print(f"  min: {w.min().item():.6f}")
    print(f"  max: {w.max().item():.6f}")
    print(f"  mean: {w.mean().item():.6f}")
    print(f"  std: {w.std().item():.6f}")
    print(f"  has NaN: {torch.isnan(w).any().item()}")
    print(f"  has Inf: {torch.isinf(w).any().item()}")
    
    print("\n✅ Model verification complete")

if __name__ == "__main__":
    verify_model()