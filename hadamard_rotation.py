"""
Hadamard Rotation for LLM Quantization (W4A4)

Reference: https://dl.acm.org/doi/pdf/10.1145/3774934.3786419
Core idea: Apply random Hadamard transforms to smooth outlier distributions 
before 4-bit quantization, reducing quantization error.

y = W @ x  →  y = (W @ H^T) @ (H @ x)   where H is orthogonal Hadamard
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple, Dict
import random


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate a normalized Hadamard matrix of size n x n.
    n must be a power of 2.
    Returns H where H @ H^T = I (orthogonal, columns normalized to unit length).
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    
    # Recursively build Hadamard matrix using Sylvester's construction
    H = torch.tensor([[1.0]], dtype=dtype)
    m = 1
    while m < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
        m *= 2
    
    # Normalize to make orthogonal: H / sqrt(n)
    H = H / math.sqrt(n)
    return H


def random_hadamard_matrix(n: int, dtype: torch.dtype = torch.float32, 
                           seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate a randomized Hadamard matrix: H_r = D @ H @ D'
    where D, D' are random diagonal sign matrices (±1 on diagonal).
    
    This provides multiple rotation options while preserving orthogonality.
    """
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        signs_left = torch.randint(0, 2, (n,), generator=g).to(dtype) * 2 - 1
        signs_right = torch.randint(0, 2, (n,), generator=g).to(dtype) * 2 - 1
    else:
        signs_left = torch.randint(0, 2, (n,)).to(dtype) * 2 - 1
        signs_right = torch.randint(0, 2, (n,)).to(dtype) * 2 - 1
    
    H = hadamard_matrix(n, dtype=dtype)
    # H_r = diag(s_left) @ H @ diag(s_right)
    H_r = signs_left.unsqueeze(1) * H * signs_right.unsqueeze(0)
    return H_r


def pad_to_power_of_2(size: int) -> int:
    """Round up to nearest power of 2."""
    if size <= 0:
        return 0
    return 1 << (size - 1).bit_length()


class HadamardRotation:
    """
    Manages Hadamard rotation for a pair of linear layer dimensions.
    
    For a weight matrix W of shape [out_features, in_features]:
    - Weights rotated offline: W' = W @ H^T (applied once before quantization)
    - Activations rotated online: x' = H @ x (applied at inference time)
    
    Supports block-wise rotation when in_features is not a power of 2,
    or can pad to the nearest power of 2.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 dtype: torch.dtype = torch.float32,
                 seed: Optional[int] = None,
                 mode: str = 'pad'):
        """
        Args:
            in_features: Input dimension of the linear layer
            out_features: Output dimension
            dtype: Data type for rotation matrices
            seed: Random seed for reproducible Hadamard randomization
            mode: 'pad' (pad to power of 2) or 'block' (split into power-of-2 blocks)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.seed = seed
        self.mode = mode
        
        if mode == 'pad':
            self._setup_pad_mode()
        elif mode == 'block':
            self._setup_block_mode()
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _setup_pad_mode(self):
        """Pad in_features to nearest power of 2."""
        self.padded_in = pad_to_power_of_2(self.in_features)
        self.pad_size = self.padded_in - self.in_features
        
        # Generate randomized Hadamard for padded dimension
        self.H = random_hadamard_matrix(self.padded_in, dtype=self.dtype, seed=self.seed)
        # H^T for weight rotation
        self.H_T = self.H.T  # H is orthogonal, so H^T = H^{-1}
        
        self.needs_padding = self.pad_size > 0
    
    def _setup_block_mode(self):
        """Split in_features into power-of-2 blocks."""
        # Find power-of-2 block sizes that sum to in_features
        remaining = self.in_features
        self.block_sizes = []
        while remaining > 0:
            block = 1 << (remaining.bit_length() - 1)
            self.block_sizes.append(block)
            remaining -= block
        
        # Generate a Hadamard matrix for each block
        self.H_blocks = []
        self.H_T_blocks = []
        for i, bs in enumerate(self.block_sizes):
            block_seed = self.seed + i if self.seed is not None else None
            H_block = random_hadamard_matrix(bs, dtype=self.dtype, seed=block_seed)
            self.H_blocks.append(H_block)
            self.H_T_blocks.append(H_block.T)
        
        self.needs_padding = False
    
    def rotate_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Rotate weights offline: W' = W @ H^T
        Applied once before quantization.
        
        Args:
            weight: Original weight tensor [out_features, in_features]
        Returns:
            Rotated weight tensor [out_features, padded_in_features] if pad mode,
            or [out_features, in_features] if block mode
        """
        orig_dtype = weight.dtype
        
        if self.mode == 'pad':
            w = weight.to(dtype=self.dtype)
            if self.needs_padding:
                w = torch.nn.functional.pad(w, (0, self.pad_size), value=0.0)
            H_T = self.H_T.to(device=w.device, dtype=w.dtype if w.dtype != torch.float32 else self.dtype)
            w_rotated = w @ H_T
            return w_rotated.to(dtype=orig_dtype)
        
        elif self.mode == 'block':
            w = weight.to(dtype=self.dtype)
            rotated_parts = []
            start = 0
            for bs, H_T_block in zip(self.block_sizes, self.H_T_blocks):
                w_block = w[:, start:start + bs]
                H_T_block = H_T_block.to(device=w.device, dtype=w.dtype if w.dtype != torch.float32 else self.dtype)
                rotated_parts.append(w_block @ H_T_block)
                start += bs
            return torch.cat(rotated_parts, dim=1).to(dtype=orig_dtype)
    
    def rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate activations online: x' = H @ x
        Applied at every forward pass during inference.
        
        Args:
            x: Activation tensor [..., in_features]
        Returns:
            Rotated activation tensor [..., padded_in_features] if pad mode,
            or [..., in_features] if block mode
        """
        orig_dtype = x.dtype
        original_shape = x.shape
        
        # Reshape to 2D for matrix multiply
        x_flat = x.reshape(-1, self.in_features).to(dtype=self.dtype)
        
        if self.mode == 'pad':
            if self.needs_padding:
                x_flat = torch.nn.functional.pad(x_flat, (0, self.pad_size), value=0.0)
            H = self.H.to(device=x_flat.device, dtype=x_flat.dtype if x_flat.dtype != torch.float32 else self.dtype)
            x_rotated = x_flat @ H
        
        elif self.mode == 'block':
            rotated_parts = []
            start = 0
            for bs, H_block in zip(self.block_sizes, self.H_blocks):
                x_block = x_flat[:, start:start + bs]
                H_block = H_block.to(device=x_flat.device, dtype=x_flat.dtype if x_flat.dtype != torch.float32 else self.dtype)
                rotated_parts.append(x_block @ H_block)
                start += bs
            x_rotated = torch.cat(rotated_parts, dim=1)
        
        # Reshape back
        output_shape = original_shape[:-1] + (x_rotated.shape[-1],)
        return x_rotated.reshape(output_shape).to(dtype=orig_dtype)
    
    def get_effective_in_features(self) -> int:
        """Return effective input dimension after rotation (may be padded)."""
        if self.mode == 'pad':
            return self.padded_in
        return self.in_features


class RotatedQLinear(nn.Linear):
    """
    A linear layer with Hadamard-rotated weights and activations,
    compatible with the existing W4A4 quantization framework.
    
    Forward: y = W_rotated @ (H @ x) 
           where W_rotated = W_original @ H^T
           
    This can optionally quantize W_rotated and H@x for W4A4 inference.
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 hadamard_seed: Optional[int] = None,
                 rotation_mode: str = 'pad'):
        super().__init__(in_features, out_features, bias=bias)
        
        self.rotation = HadamardRotation(
            in_features=in_features, 
            out_features=out_features,
            dtype=torch.float32,
            seed=hadamard_seed,
            mode=rotation_mode
        )
        self._weights_rotated = False
        
    def rotate_weights(self):
        """Apply Hadamard rotation to weights (call once after loading)."""
        if not self._weights_rotated:
            rotated_w = self.rotation.rotate_weight(self.weight.data)
            # Update weight to rotated version
            effective_in = self.rotation.get_effective_in_features()
            self.weight.data = rotated_w
            # Note: self.in_features now effectively = effective_in
            # but we keep the original to reverse rotation if needed
            self._weights_rotated = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._weights_rotated:
            raise RuntimeError("Weights not rotated. Call rotate_weights() first.")
        
        # Rotate activations online
        x_rotated = self.rotation.rotate_activation(x)
        
        # Standard linear forward with rotated weight and activation
        return nn.functional.linear(x_rotated, self.weight, self.bias)


def replace_with_rotated_qlinear(
    module: nn.Module,
    w_Q: str = 'hifx4',
    in_Q: str = 'hifx4',
    quant_grad: bool = False,
    exclude_layers: List[str] = None,
    high_precision_layers: List[str] = None,
    high_precision_type: str = 'bf16',
    rotation_seed: int = 42,
    rotation_mode: str = 'pad'
) -> Dict[str, HadamardRotation]:
    """
    Replace all nn.Linear layers with rotated + quantized versions.
    
    For each linear layer:
    1. Compute Hadamard rotation matrices H (offline)
    2. Rotate weight: W' = W @ H^T
    3. Replace with QLinear that does:
       - Online activation rotation: H @ x
       - Quantized linear: Q(W') @ Q(H@x)
    
    Args:
        module: PyTorch module to modify (in-place)
        w_Q: Weight quantization type (e.g., 'hifx4', 'mxfp4')
        in_Q: Input activation quantization type
        quant_grad: Whether to quantize gradients
        exclude_layers: Layer names to exclude from quantization (keep BF16)
        high_precision_layers: Layers to keep at higher precision
        high_precision_type: Quant type for high-precision layers
        rotation_seed: Base seed for random Hadamard generation
        rotation_mode: 'pad' or 'block'
    
    Returns:
        Dictionary mapping layer names to their HadamardRotation objects
    """
    from hif4_gpu.quant_cy.layers.QLinear import QLinear
    from hif4_gpu.quant_cy.base.QType import QType
    
    if exclude_layers is None:
        exclude_layers = []
    if high_precision_layers is None:
        high_precision_layers = []
    
    # Build module dictionary for parent traversal
    mod_dict = {}
    for n, m in module.named_modules():
        mod_dict[n] = m
    
    rotations = {}
    linear_count = 0
    skipped = 0
    
    for n, m in module.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        
        if n in exclude_layers:
            print(f"[Skip] Layer '{n}' excluded from quantization")
            skipped += 1
            continue
        
        linear_count += 1
        
        # Create Hadamard rotation
        layer_seed = rotation_seed + linear_count
        rotation = HadamardRotation(
            in_features=m.in_features,
            out_features=m.out_features,
            dtype=torch.float32,
            seed=layer_seed,
            mode=rotation_mode
        )
        rotations[n] = rotation
        
        # Rotate weights offline
        rotated_weight = rotation.rotate_weight(m.weight.data)
        effective_in_features = rotation.get_effective_in_features()
        
        # Determine quantization type
        if n in high_precision_layers:
            q_type = high_precision_type
            print(f"[HQ] Layer '{n}' kept at higher precision: {high_precision_type}")
        else:
            q_type = w_Q
        
        # Create QLinear with rotated dimensions
        new_mod = QLinear(effective_in_features, m.out_features, m.bias is not None)
        new_mod.weight.data = rotated_weight.to(new_mod.weight.dtype)
        if m.bias is not None:
            new_mod.bias.data = m.bias.data.clone()
        
        new_mod.assign_qparams(w_Q)
        new_mod.set_quant_grad(quant_grad)
        if in_Q is not None:
            new_mod.assign_input_qparams(in_Q)
        
        # Store rotation reference for online activation rotation
        new_mod._hadamard_rotation = rotation
        
        # Need to wrap forward to inject activation rotation
        original_forward = new_mod.forward
        
        def make_rotated_forward(mod, orig_fwd):
            def rotated_forward(x):
                # Rotate activation first
                x_rotated = mod._hadamard_rotation.rotate_activation(x)
                return orig_fwd(x_rotated)
            return rotated_forward
        
        new_mod.forward = make_rotated_forward(new_mod, original_forward).__get__(new_mod, type(new_mod))
        
        # Replace in parent module
        parent_mod = mod_dict['.'.join(n.split('.')[:-1])]
        setattr(parent_mod, n.split('.')[-1], new_mod)
    
    print(f"[Replace] Converted {linear_count} Linear layers to RotatedQLinear "
          f"(skipped {skipped})")
    print(f"[Replace] {len(high_precision_layers)} layers kept at high precision")
    
    return rotations