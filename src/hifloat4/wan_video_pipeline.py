#!/usr/bin/env python3
"""Shared Wan 2.2 video pipeline assembly helpers."""

import gc
import json
import os
from typing import List, Optional

import numpy as np
import torch

from hifloat4.wan_model_utils import (
    _build_legacy_wan_init_kwargs,
    _is_legacy_wan_config,
    _load_legacy_wan_weights,
    _load_wan_weight_index,
    _read_json,
    load_wan_model,
)
from hifloat4.wan_rotation import block_diagonal_hadamard, register_rotation_hooks


def _resolve_video_path(raw_path: str) -> Optional[str]:
    if os.path.isfile(raw_path):
        return raw_path
    remapped = raw_path.replace("/home/datasets/OpenS2V-5M/", "datasets/")
    if os.path.isfile(remapped):
        return remapped
    remapped2 = raw_path.replace("/home/datasets/", "datasets/")
    if os.path.isfile(remapped2):
        return remapped2
    return None


def load_dataset(dataset_path: str, max_samples: int = -1) -> List[dict]:
    if not os.path.exists(dataset_path):
        print(f"[Dataset] {dataset_path} not found")
        return []
    with open(dataset_path, "r") as f:
        data = json.load(f)
    samples = data if isinstance(data, list) else data.get("samples", [])
    valid = []
    for s in samples:
        raw_path = s.get("path") or s.get("video_path")
        if not raw_path:
            continue
        resolved = _resolve_video_path(raw_path)
        if resolved:
            s["resolved_path"] = resolved
            valid.append(s)
        if max_samples > 0 and len(valid) >= max_samples:
            break
    print(f"[Dataset] {len(valid)} valid samples with accessible videos "
          f"(from {len(samples)} total)")
    return valid


def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        return vr[0].asnumpy()
    except Exception as e:
        print(f"  [Warn] Failed to extract frame from {video_path}: {e}")
        return None


def save_video_frames_as_mp4(frames: np.ndarray, output_path: str, fps: int = 16):
    """Save video frames as MP4 with proper uint8 conversion.

    The Wan pipeline outputs float32 frames in [0, 1] range via
    ``postprocess_video``.  We convert to uint8 before saving to avoid
    lossy-conversion warnings and ensure maximum encoded quality.
    """
    # Convert float32 [0,1] → uint8 [0,255] if needed
    if frames.dtype in (np.float32, np.float64):
        frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)

    try:
        import imageio.v3 as iio
        iio.imwrite(output_path, frames, fps=fps, codec="libx264",
                     quality=None, pixelformat="yuv420p",
                     output_params=["-crf", "0"])
    except Exception:
        os.makedirs(output_path.replace(".mp4", "_frames"), exist_ok=True)
        from PIL import Image
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(
                os.path.join(output_path.replace(".mp4", "_frames"), f"{i:04d}.png"))


def _convert_wan_t5_to_diffusers(state_dict):
    """Convert Wan T5 checkpoint keys to UMT5EncoderModel state dict keys.

    Wan uses a custom T5 encoder with keys like ``blocks.{i}.attn.q.weight``
    while HuggingFace UMT5EncoderModel expects keys like
    ``encoder.block.{i}.layer.0.SelfAttention.q.weight``.
    """
    mapping = {}
    for wan_key in state_dict:
        new_key = wan_key

        if new_key == "token_embedding.weight":
            new_key = "encoder.embed_tokens.weight"

        elif new_key == "norm.weight":
            new_key = "encoder.final_layer_norm.weight"

        elif new_key.startswith("blocks."):
            # blocks.{i}.xxx → encoder.block.{i}.xxx
            new_key = new_key.replace("blocks.", "encoder.block.", 1)

            # pos_embedding.embedding → layer.0.SelfAttention.relative_attention_bias
            new_key = new_key.replace(".pos_embedding.embedding.weight",
                                      ".layer.0.SelfAttention.relative_attention_bias.weight")

            # norm1 → layer.0.layer_norm
            new_key = new_key.replace(".norm1.", ".layer.0.layer_norm.")

            # norm2 → layer.1.layer_norm
            new_key = new_key.replace(".norm2.", ".layer.1.layer_norm.")

            # attn.q → layer.0.SelfAttention.q
            new_key = new_key.replace(".attn.q.", ".layer.0.SelfAttention.q.")
            new_key = new_key.replace(".attn.k.", ".layer.0.SelfAttention.k.")
            new_key = new_key.replace(".attn.v.", ".layer.0.SelfAttention.v.")
            new_key = new_key.replace(".attn.o.", ".layer.0.SelfAttention.o.")

            # ffn.gate.0 → layer.1.DenseReluDense.wi_1
            new_key = new_key.replace(".ffn.gate.0.", ".layer.1.DenseReluDense.wi_1.")

            # ffn.fc1 → layer.1.DenseReluDense.wi_0
            new_key = new_key.replace(".ffn.fc1.", ".layer.1.DenseReluDense.wi_0.")

            # ffn.fc2 → layer.1.DenseReluDense.wo
            new_key = new_key.replace(".ffn.fc2.", ".layer.1.DenseReluDense.wo.")

        elif new_key.startswith("pos_embedding"):
            # Standalone pos_embedding (not inside blocks) — skip
            continue

        mapping[wan_key] = new_key

    converted = {}
    for wan_key, tensor in state_dict.items():
        if wan_key in mapping:
            converted[mapping[wan_key]] = tensor

    # UMT5EncoderModel expects shared.weight (weight-tied with embed_tokens).
    # The Wan checkpoint only has token_embedding.weight, so we add shared.weight
    # as an alias to satisfy load_state_dict with strict=False.
    if "encoder.embed_tokens.weight" in converted and "shared.weight" not in converted:
        converted["shared.weight"] = converted["encoder.embed_tokens.weight"]

    return converted


def _load_t5_encoder(model_path: str, device: str, dtype: torch.dtype):
    from transformers import UMT5EncoderModel, UMT5Config, AutoTokenizer

    tok_dir = os.path.join(model_path, "google", "umt5-xxl")
    assert os.path.isfile(os.path.join(tok_dir, "spiece.model")), \
        f"UMT5 tokenizer not found at {tok_dir}"
    print("[Pipeline] Loading UMT5 tokenizer from local copy...")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    t5_pth = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    assert os.path.isfile(t5_pth), f"T5 weights not found: {t5_pth}"

    print(f"[Pipeline] Loading T5 encoder from {t5_pth} ...")
    config = UMT5Config(
        vocab_size=256384, d_model=4096, d_kv=64, d_ff=10240,
        num_layers=24, num_heads=64, is_encoder_decoder=False, use_cache=False,
    )
    text_encoder = UMT5EncoderModel(config).to(device=device, dtype=dtype)
    sd = torch.load(t5_pth, map_location=device, weights_only=True)
    converted = _convert_wan_t5_to_diffusers(sd)
    load_result = text_encoder.load_state_dict(converted, strict=False)
    if load_result.missing_keys:
        print(f"  [T5] Missing keys: {len(load_result.missing_keys)}")
        for k in load_result.missing_keys[:10]:
            print(f"    {k}")
    if load_result.unexpected_keys:
        print(f"  [T5] Unexpected keys: {len(load_result.unexpected_keys)}")
        for k in load_result.unexpected_keys[:10]:
            print(f"    {k}")
    text_encoder.eval()
    del sd, converted
    gc.collect()
    return tokenizer, text_encoder


def _load_vae(model_path: str, device: str, dtype: torch.dtype):
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers

    vae_pth = os.path.join(model_path, "Wan2.1_VAE.pth")
    assert os.path.isfile(vae_pth), f"VAE weights not found: {vae_pth}"

    print(f"[Pipeline] Loading VAE from {vae_pth} ...")
    vae_sd = torch.load(vae_pth, map_location="cpu", weights_only=True)
    converted_sd = convert_wan_vae_to_diffusers(dict(vae_sd))
    del vae_sd
    gc.collect()

    vae = AutoencoderKLWan()
    load_result = vae.load_state_dict(converted_sd, strict=False)
    if load_result.missing_keys:
        print(f"  [VAE] Missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"  [VAE] Unexpected keys: {len(load_result.unexpected_keys)}")

    del converted_sd
    gc.collect()

    vae = vae.to(device=device, dtype=dtype)
    vae.eval()
    return vae


def _load_bf16_transformer(model_path: str, device: str, dtype: torch.dtype,
                            subdir: str = "low_noise_model"):
    load_dir = os.path.join(model_path, subdir)
    transformer = load_wan_model(load_dir, device=device, dtype=dtype)
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()

    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"  [{subdir}] Loaded: {total_params/1e9:.2f}B params in BF16")
    gc.collect()
    return transformer


def _load_quantized_transformer(model_path: str, quantized_path: str,
                                 device: str, dtype: torch.dtype,
                                 subdir: str = "high_noise_model"):
    from hif4_gpu.quant_cy.utils.utils import replace_linear

    meta_path = os.path.join(quantized_path, "quantization_metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    quant_type = metadata.get("quant_type", "hifx4")
    rotation_mode = metadata.get("rotation_mode", "none")
    rotation_seed = int(metadata.get("rotation_seed", 42))
    use_rotation = rotation_mode != "none"
    # Restore the same high-precision layers that were excluded during quantization.
    # If the metadata doesn't have this field (old checkpoints), default to empty.
    high_precision_layers = metadata.get("high_precision_layers", [])

    print(f"[Pipeline] Loading transformer ({subdir}) from {model_path} ...")
    if subdir == "low_noise_model":
        load_dir = os.path.join(model_path, "low_noise_model")
        raw_cfg = _read_json(os.path.join(load_dir, "config.json"))
        if _is_legacy_wan_config(raw_cfg):
            from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
            weight_map = _load_wan_weight_index(load_dir)
            init_kwargs = _build_legacy_wan_init_kwargs(load_dir, raw_cfg, weight_map)
            transformer = WanTransformer3DModel(**init_kwargs)
            _load_legacy_wan_weights(transformer, load_dir, weight_map)
        else:
            from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
            transformer = WanTransformer3DModel.from_pretrained(
                load_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
        transformer = transformer.to(device=device, dtype=dtype)
        transformer.eval()
    else:
        transformer = load_wan_model(model_path, device=device, dtype=dtype)

    print(f"[Pipeline] Applying {quant_type} quantization structure to {subdir}...")
    if high_precision_layers:
        print(f"[Pipeline] Excluding {len(high_precision_layers)} high-precision layers:")
        for name in high_precision_layers:
            print(f"  Keeping BF16: {name}")
    replace_linear(transformer, w_Q=quant_type, in_Q=quant_type,
                   quant_grad=False, exclude_layers=high_precision_layers)

    ckpt_path = os.path.join(quantized_path, "quantized_transformer.pt")
    print(f"[Pipeline] Loading weights from {ckpt_path} ...")
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    load_result = transformer.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"  [{subdir}] Missing keys: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"  [{subdir}] Unexpected keys: {len(load_result.unexpected_keys)}")

    transformer.eval()

    if use_rotation and subdir == "high_noise_model":
        H_dict = {}
        in_features_set = set()
        for _name, module in transformer.named_modules():
            if hasattr(module, "in_features"):
                in_features_set.add(module.in_features)
        for in_f in in_features_set:
            H_dict[in_f] = block_diagonal_hadamard(
                in_f, seed=rotation_seed + (in_f * 9973) % 10007)
        hooks = register_rotation_hooks(transformer, H_dict)
        print(f"[Pipeline] Registered {len(hooks)} FWHT rotation hooks on {subdir}")

    del payload, state_dict
    gc.collect()
    return transformer


def build_bf16_pipeline(model_path: str, device: str, dtype: torch.dtype):
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    transformer = _load_bf16_transformer(model_path, device, dtype, "high_noise_model")
    transformer_2 = _load_bf16_transformer(model_path, device, dtype, "low_noise_model")
    tokenizer, text_encoder = _load_t5_encoder(model_path, device, dtype)
    vae = _load_vae(model_path, device, dtype)
    # NOTE: Wan 2.2 I2V does NOT use CLIP image encoder!
    # model_index.json declares image_encoder: [null, null]
    # The pipeline checks transformer.config.image_dim and skips CLIP when it's None.
    # Passing None saves ~1.6GB VRAM.

    # Wan 2.2 I2V-A14B official scheduler config from HuggingFace
    # Uses UniPCMultistepScheduler with flow_shift=3.0
    scheduler = UniPCMultistepScheduler.from_config({
        "_class_name": "UniPCMultistepScheduler",
        "_diffusers_version": "0.35.0.dev0",
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "disable_corrector": [],
        "dynamic_thresholding_ratio": 0.995,
        "final_sigmas_type": "zero",
        "flow_shift": 3.0,
        "lower_order_final": True,
        "num_train_timesteps": 1000,
        "predict_x0": True,
        "prediction_type": "flow_prediction",
        "rescale_betas_zero_snr": False,
        "sample_max_value": 1.0,
        "solver_order": 2,
        "solver_p": None,
        "solver_type": "bh2",
        "steps_offset": 0,
        "thresholding": False,
        "time_shift_type": "exponential",
        "timestep_spacing": "linspace",
        "trained_betas": None,
        "use_beta_sigmas": False,
        "use_dynamic_shifting": False,
        "use_exponential_sigmas": False,
        "use_flow_sigmas": True,
        "use_karras_sigmas": False,
    })

    pipe = WanImageToVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
        scheduler=scheduler, image_processor=None,
        image_encoder=None, transformer=transformer,
        transformer_2=transformer_2, boundary_ratio=0.9, expand_timesteps=False,
    )
    pipe = pipe.to(device)
    print("[Pipeline] BF16 baseline pipeline assembled (no quantization, no CLIP)")
    return pipe


def build_quantized_pipeline(model_path: str, quantized_path: str,
                              device: str, dtype: torch.dtype):
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    transformer = _load_quantized_transformer(
        model_path, quantized_path, device, dtype, "high_noise_model")
    print("[Pipeline] Loading low_noise_model (transformer_2) in BF16...")
    transformer_2 = _load_bf16_transformer(model_path, device, dtype, "low_noise_model")
    tokenizer, text_encoder = _load_t5_encoder(model_path, device, dtype)
    vae = _load_vae(model_path, device, dtype)
    # NOTE: Wan 2.2 I2V does NOT use CLIP (image_encoder: [null, null] in model_index.json)

    # Wan 2.2 I2V-A14B official scheduler config from HuggingFace
    scheduler = UniPCMultistepScheduler.from_config({
        "_class_name": "UniPCMultistepScheduler",
        "_diffusers_version": "0.35.0.dev0",
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "disable_corrector": [],
        "dynamic_thresholding_ratio": 0.995,
        "final_sigmas_type": "zero",
        "flow_shift": 3.0,
        "lower_order_final": True,
        "num_train_timesteps": 1000,
        "predict_x0": True,
        "prediction_type": "flow_prediction",
        "rescale_betas_zero_snr": False,
        "sample_max_value": 1.0,
        "solver_order": 2,
        "solver_p": None,
        "solver_type": "bh2",
        "steps_offset": 0,
        "thresholding": False,
        "time_shift_type": "exponential",
        "timestep_spacing": "linspace",
        "trained_betas": None,
        "use_beta_sigmas": False,
        "use_dynamic_shifting": False,
        "use_exponential_sigmas": False,
        "use_flow_sigmas": True,
        "use_karras_sigmas": False,
    })

    pipe = WanImageToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        image_processor=None,
        image_encoder=None,
        transformer=transformer,
        transformer_2=transformer_2,
        boundary_ratio=0.9,
        expand_timesteps=False,
    )
    pipe = pipe.to(device)
    print("[Pipeline] Quantized I2V pipeline assembled successfully!")
    print(f"  transformer  (high_noise): quantized W4A4")
    print(f"  transformer_2 (low_noise): BF16 (full precision)")
    print(f"  boundary_ratio=0.9, expand_timesteps=False (14B cat mode)")
    print(f"  CLIP: not loaded (Wan 2.2 I2V doesn't use it)")
    return pipe
