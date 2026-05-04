#!/usr/bin/env python3
"""Hadamard rotation helpers for Wan quantization."""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def hadamard_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    assert dim > 0 and (dim & (dim - 1)) == 0, f"dim={dim} must be a power of 2"
    rng = torch.Generator()
    rng.manual_seed(seed)

    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32)
    n = 2
    while n < dim:
        H = torch.kron(H, torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32))
        n *= 2
    signs = torch.randint(0, 2, (dim,), generator=rng).float() * 2 - 1
    H = H * signs.unsqueeze(1)
    H = H / math.sqrt(dim)
    return H


def block_diagonal_hadamard(dim: int, seed: int = 42) -> torch.Tensor:
    assert dim > 0, f"dim must be positive, got {dim}"
    blocks: List[int] = []
    remaining = dim
    while remaining > 0:
        blk = 1 << (remaining.bit_length() - 1)
        blocks.append(blk)
        remaining -= blk

    parts: List[torch.Tensor] = []
    for idx, blk in enumerate(blocks):
        parts.append(hadamard_matrix(blk, seed=seed + idx * 137))

    if len(parts) == 1:
        return parts[0]

    H_full = torch.zeros(dim, dim, dtype=torch.float32)
    offset = 0
    for blk_mat in parts:
        sz = blk_mat.shape[0]
        H_full[offset:offset + sz, offset:offset + sz] = blk_mat
        offset += sz
    return H_full


def rotate_linear_weights(model: nn.Module, mode: str = "pad", seed: int = 42) -> Dict[int, torch.Tensor]:
    del mode
    H_dict: Dict[int, torch.Tensor] = {}
    rotated_count = 0

    for _name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        in_f = module.in_features
        if in_f not in H_dict:
            H_dict[in_f] = block_diagonal_hadamard(in_f, seed=seed + (in_f * 9973) % 10007)

        H = H_dict[in_f]
        W = module.weight.data
        H_dev = H.to(device=W.device, dtype=torch.float32)
        W_rotated = (W.float() @ H_dev.T).to(W.dtype)
        module.weight.data = W_rotated.contiguous()
        rotated_count += 1

    print(f"[Rotation] Rotated {rotated_count} Linear layers (mode=block-diagonal)")
    return H_dict


def fwht(x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    n = original_shape[-1]
    assert (n & (n - 1)) == 0, f"FWHT requires power-of-2 last dim, got {n}"
    x = x.reshape(-1, n).contiguous()

    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.view(-1, n)
        h *= 2

    return x.view(original_shape)


def register_rotation_hooks(model: nn.Module, H_dict: Dict[int, torch.Tensor]) -> List:
    hooks: List = []
    dim_params: Dict[int, Tuple[List[int], List[torch.Tensor]]] = {}

    for name, module in model.named_modules():
        # After replace_linear(), nn.Linear becomes QLinear (not an nn.Linear subclass).
        # Use hasattr to detect any module with in_features that participates in the forward pass.
        if not isinstance(module, nn.Linear) and not hasattr(module, 'in_features'):
            continue
        in_f = getattr(module, 'in_features', None)
        if in_f is None or in_f not in H_dict:
            continue

        if in_f not in dim_params:
            H = H_dict[in_f]
            n = H.shape[0]
            blocks: List[int] = []
            remaining = n
            while remaining > 0:
                blk = 1 << (remaining.bit_length() - 1)
                blocks.append(blk)
                remaining -= blk
            signs_list: List[torch.Tensor] = []
            offset = 0
            for blk in blocks:
                H_block = H[offset:offset + blk, offset:offset + blk]
                signs = (H_block[:, 0] * math.sqrt(blk)).round().clamp(-1, 1)
                signs_list.append(signs)
                offset += blk
            dim_params[in_f] = (blocks, signs_list)

        blocks, signs_list = dim_params[in_f]

        def _make_fast_hook(blks: List[int], sgns: List[torch.Tensor]):
            @torch.no_grad()
            def _rotation_hook(_module, args):
                x = args[0]
                orig_dtype = x.dtype
                x_f = x.float().contiguous()

                if len(blks) == 1:
                    bs = blks[0]
                    s = sgns[0].to(device=x.device)
                    x_h = fwht(x_f)
                    return (x_h * s / math.sqrt(bs)).to(orig_dtype)
                parts = []
                offset = 0
                for i, bs in enumerate(blks):
                    xb = x_f[..., offset:offset + bs].contiguous()
                    s = sgns[i].to(device=x.device)
                    xh = fwht(xb)
                    parts.append(xh * s / math.sqrt(bs))
                    offset += bs
                return torch.cat(parts, dim=-1).to(orig_dtype)
            return _rotation_hook

        handle = module.register_forward_pre_hook(_make_fast_hook(blocks, signs_list))
        hooks.append((name, handle))

    print(f"[Rotation] Registered {len(hooks)} FAST rotation hooks (FWHT, O(n log n))")
    return hooks
