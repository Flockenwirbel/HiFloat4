# 竞赛达标验证指南

## 竞赛门槛（Baseline Requirement）

> **HiF4 量化模型的 VBench 平均精度损失 < 0.5%（对比 BF16 baseline）**
> 
> 不满足此门槛的提交将不进入排名。

## VBench 评估的 5 个维度

| 维度 | 说明 |
|------|------|
| Imaging Quality | 图像质量（musiq_spaq） |
| Aesthetic Quality | 美学质量（CLIP ViT-L/14） |
| Overall Consistency | 文本-视频一致性（CLIP ViT-B/32） |
| Subject Consistency | 主体一致性（DINO） |
| Motion Smoothness | 运动平滑度（AMT-S） |

## 达标判定公式

```
loss_ratio = (BF16_score - Quantized_score) / BF16_score * 100%
```

**每个维度的 loss_ratio 都应 < 0.5%，平均也应 < 0.5%。**

---

## 步骤 1：生成测试视频（2~3 个）

需要**同时**生成量化版和 BF16 baseline 版本的视频，用相同输入才能对比。

### 1a. 量化版视频（3 个样本）

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=80G --time=8:00:00 \
  --job-name=wan_gen --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 generate_videos.py \
    --quantized-path ./quantized_wan_output \
    --num-videos 3 \
    --num-inference-steps 50'
```

### 1b. BF16 Baseline 视频（相同 3 个样本）

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=80G --time=8:00:00 \
  --job-name=wan_bf16 --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 generate_videos.py \
    --bf16-baseline \
    --num-videos 3 \
    --num-inference-steps 50'
```

> **注意**：默认 resolution=720p, num-frames=61 已更新，可以省略这两个参数。

---

## 步骤 2：VBench 评估

### 2a. 评估量化版

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --job-name=vbench_q --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 scripts/evaluate_vbench.py \
    --videos-dir ./quantized_wan_output/generated_videos_v2'
```

### 2b. 评估 BF16 baseline

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --job-name=vbench_bf16 --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 scripts/evaluate_vbench.py \
    --videos-dir ./generated_videos_bf16_baseline'
```

---

## 步骤 3：对比结果

评估完成后，查看结果：

```bash
# 量化版结果
cat ./quantized_wan_output/generated_videos_test/vbench_eval/*_eval_results.json

# BF16 baseline 结果
cat ./quantized_wan_output/generated_videos_bf16_test/vbench_eval/*_eval_results.json
```

### 达标判定表

| 维度 | BF16 分数 | 量化分数 | 损失 (%) | 达标 (<0.5%) |
|------|-----------|----------|----------|-------------|
| imaging_quality | _填入_ | _填入_ | _计算_ | ✅/❌ |
| aesthetic_quality | _填入_ | _填入_ | _计算_ | ✅/❌ |
| overall_consistency | _填入_ | _填入_ | _计算_ | ✅/❌ |
| subject_consistency | _填入_ | _填入_ | _计算_ | ✅/❌ |
| motion_smoothness | _填入_ | _填入_ | _计算_ | ✅/❌ |
| **平均** | | | | ✅/❌ |

---

## 常用 srun 命令（均指定 1×H100，实时输出）

### 量化

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=80G --time=02:00:00 \
  --job-name=wan_quant --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 quantize_wan.py \
    --model-path /home/dataset/Wan2.2-I2V-A14B \
    --output-dir ./quantized_wan_output \
    --quant-type hifx4 \
    --max-high-precision-layers 2'
```

### 生成视频（量化版）

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=80G --time=8:00:00 \
  --job-name=wan_gen --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 generate_videos.py \
    --quantized-path ./quantized_wan_output \
    --num-videos 3 \
    --num-inference-steps 50'
```

### 生成视频（BF16 baseline）

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=80G --time=8:00:00 \
  --job-name=wan_bf16 --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 generate_videos.py \
    --bf16-baseline \
    --num-videos 3 \
    --num-inference-steps 50'
```

### VBench 评估

```bash
srun --partition=Star --gres=gpu:H100:1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --job-name=vbench --pty bash -c 'source /home/liujh/miniconda3/etc/profile.d/conda.sh && conda activate HiFloat4 && \
  python3 scripts/evaluate_vbench.py --videos-dir <VIDEOS_DIR>'
```

> **提示**：`--pty` 提供交互式终端，可以实时看到 Python 输出。去掉 `--pty` 则后台运行。
> sbatch 脚本仍保留在 `scripts/` 目录下供批量提交使用。

---

## 重要提醒

1. **样本数量**：2-3 个视频只是快速验证，正式提交需要在完整 OpenS2V-5M 数据集上评估
2. **分辨率必须 720p**（720×1280），帧数 61 —— 已更新为默认值
3. **高精度层最多 2 层**（HiF4）—— 当前 2 层，已满足
4. **VBench 精度损失 < 0.5%** 是硬性门槛，不达标直接淘汰
5. 如果损失过大，可以尝试：
   - 调整 sensitivity analysis 选出最关键的 2 层保留
   - 尝试不同的 rotation_seed
   - 调整 guidance_scale（当前 5.0）
   - 增加 num_inference_steps（当前 50）