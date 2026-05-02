#!/usr/bin/env python3
import os
import sys
import torch
from PIL import Image
import numpy as np

model_path = sys.argv[1] if len(sys.argv) > 1 else '/home/dataset/Wan2.2-I2V-A14B'
img_path = os.path.join(model_path, 'examples', 'i2v_input.JPG')
if not os.path.isfile(img_path):
    # try alternative
    img_path = None
    for root, _, files in os.walk(model_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                break
        if img_path:
            break
if img_path is None:
    print('No example image found in model dir; aborting')
    sys.exit(1)

print('Using image:', img_path)

from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

# load VAE
vae_pth = os.path.join(model_path, 'Wan2.1_VAE.pth')
assert os.path.isfile(vae_pth), vae_pth
ckpt = torch.load(vae_pth, map_location='cpu')
conv = convert_wan_vae_to_diffusers(dict(ckpt))
vae = AutoencoderKLWan().to(device=device, dtype=dtype)
vae.load_state_dict(conv, strict=False)
vae.eval()

print('VAE config z_dim:', vae.config.z_dim)
print('VAE latents_mean (config):', vae.config.latents_mean)
print('VAE latents_std (config):', vae.config.latents_std)

# load CLIP
clip_dir = '/home/dataset/clip-vit-large-patch14'
assert os.path.isdir(clip_dir), clip_dir
image_encoder = CLIPVisionModel.from_pretrained(clip_dir)
image_processor = CLIPImageProcessor.from_pretrained(clip_dir)
image_encoder = image_encoder.to(device)
image_encoder.eval()

# prepare image
img = Image.open(img_path).convert('RGB')
inputs = image_processor(images=img, return_tensors='pt')
pixel_values = inputs['pixel_values'].to(device)

# run VAE encode via pipeline code path: build a video condition of single frame
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents

vae_input = pixel_values.unsqueeze(2)  # [B,C,1,H,W]
with torch.no_grad():
    enc_out = vae.encode(vae_input)
    posterior = enc_out.latent_dist
    lat = posterior.mode()  # shape [B, z_dim, frames, h, w]

lat_cpu = lat.detach().cpu().numpy()
zdim = lat_cpu.shape[1]
print('encoded lat shape', lat_cpu.shape)

per_ch_mean = lat_cpu.mean(axis=(0,2,3,4))
per_ch_std = lat_cpu.std(axis=(0,2,3,4))
per_ch_min = lat_cpu.min(axis=(0,2,3,4))
per_ch_max = lat_cpu.max(axis=(0,2,3,4))

print('per-channel mean (encoded):')
for i in range(zdim):
    print(f'{i}: mean={per_ch_mean[i]:.4f}, std={per_ch_std[i]:.4f}, min={per_ch_min[i]:.4f}, max={per_ch_max[i]:.4f}')

print('\nComparison to VAE config:')
for i in range(zdim):
    cfg_mean = vae.config.latents_mean[i]
    cfg_std = vae.config.latents_std[i]
    print(f'{i}: cfg_mean={cfg_mean:.4f}, cfg_std={cfg_std:.4f}, enc_mean={per_ch_mean[i]:.4f}, enc_std={per_ch_std[i]:.4f}')

print('\nDone')
