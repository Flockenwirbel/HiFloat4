#!/usr/bin/env python3
"""Generate video with first frame protected from denoising."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from PIL import Image
from hifloat4.wan_video_pipeline import build_bf16_pipeline


def prepare_protected_latents(pipe, image, num_frames=81, device="cuda", dtype=torch.bfloat16):
    """Prepare latents where first frame is encoded from input image, rest is noise."""
    from diffusers.utils.torch_utils import randn_tensor
    
    height, width = 480, 832
    num_channels_latents = 16
    
    vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
    vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
    
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial
    
    # Create latents with noise
    shape = (1, num_channels_latents, num_latent_frames, latent_height, latent_width)
    latents = randn_tensor(shape, device=device, dtype=dtype)
    
    # Encode input image to latent
    # image should be [B, C, H, W]
    image_input = image.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    image_input = image_input.to(device=device, dtype=pipe.vae.dtype)
    
    # Pad to num_frames
    video_condition = torch.cat(
        [image_input, image_input.new_zeros(1, 3, num_frames - 1, image_input.shape[-2], image_input.shape[-1])],
        dim=2
    )
    
    # Encode
    with torch.no_grad():
        latent_condition = pipe.vae.encode(video_condition).latent_dist.sample()
        latent_condition = latent_condition * pipe.vae.config.scaling_factor
    
    # Normalize
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latent_condition = (latent_condition - latents_mean) * latents_std
    
    # Replace first frame latent with encoded image
    latents[:, :, 0] = latent_condition[:, :, 0]
    
    return latents, latent_condition


def main():
    device = "cuda"
    dtype = torch.bfloat16
    model_path = "/home/dataset/Wan2.2-I2V-A14B"
    
    print("Building pipeline...")
    pipe = build_bf16_pipeline(model_path, device, dtype)
    
    # Load image
    from hifloat4.wan_video_pipeline import extract_first_frame
    video_path = "datasets/total_part2/part_180/23_OVVCrZx8/video.mp4"  # example
    frame = extract_first_frame(video_path)
    image = Image.fromarray(frame).resize((832, 480))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    print("Preparing protected latents...")
    latents, condition = prepare_protected_latents(pipe, image_tensor, device=device, dtype=dtype)
    
    print(f"Latents shape: {latents.shape}")
    print(f"First frame latent mean: {latents[0, :, 0].mean().item():.4f}")
    print(f"Other frames latent mean: {latents[0, :, 1:].mean().item():.4f}")
    
    # This is a test - actual generation would use these latents
    print("Test complete. To generate, pass latents=latents to pipe()")


if __name__ == "__main__":
    main()