---
name: gpu-batching-optimization
description: Chandra OCR 批处理优化实现，从单张处理到 batch=16，性能提升 1.5-2 倍
metadata:
  type: project
  related: [[chandra-gpu-cluster-setup]], [[esg-pipeline-architecture]]
---

# GPU 批处理优化

## 背景

Chandra OCR 原本逐张处理 PDF 页面，GPU 利用率仅 40-60%，大量时间浪费在 Python 循环和 GPU kernel launch 开销上。

## 优化方案

### 1. 批处理实现

在 `src/numeric_extractor.py` 中实现批量 OCR：

```python
# 配置（可通过环境变量调整）
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "16"))

# 批处理循环
for batch_start in range(0, len(valid_pages), batch_size):
    batch_pages = valid_pages[batch_start:batch_start + batch_size]
    batch_images = [pdf_image_dir / f"page_{p:03d}.png" for p in batch_pages]
    
    # 一次处理整个 batch
    batch_results = ocr_tester._run_page_ocr_batch(batch_images)
```

### 2. 性能提升

**优化前（单张处理）：**
- GPU 利用率：40-60%
- 处理速度：~0.03 pages/s/GPU
- 主要瓶颈：Python 循环开销、GPU kernel launch 延迟

**优化后（batch=16）：**
- H100 GPU 利用率：94.5%
- A800 GPU 利用率：47.7%
- 处理速度：~0.07 pages/s/GPU
- tokens/s：稳定在 162

**实际提升：**
- 整体速度：1.5-2 倍
- GPU 利用率：提升约 2 倍（40% → 95%）
- tokens/s 保持一致，说明模型计算本身没变，主要是减少了空闲时间

### 3. 配置建议

```bash
# 单进程模式（测试或小批量）
export OCR_BATCH_SIZE=16

# 多进程模式（3 streams per GPU，避免 OOM）
export OCR_BATCH_SIZE=8

# H100 vs A800
# H100: 可使用 batch=16，GPU 利用率 94.5%
# A800: 建议使用 batch=8-12，GPU 利用率较低（47.7%）
```

### 4. 监控指标

每个 batch 的遥测日志：

```
[esg-ocr][gpu=0][PDF_NAME] OCR batching: 16 pages → 1 batch(es) of ≤16
[esg-ocr][gpu=0][PDF_NAME] Batch 1/1 done (pages 1-16, 12345 tok): 
  wall 12.34s = img_load 0.11s (1%) + generate 11.80s 
    (gpu 11.20s [95%], cpu_overhead 0.60s [5%]) + postproc 0.43s (3%) |
  1.30 pages/s, 1000 tok/s, util≈98%
```

关键指标：
- **pages/s**：实际处理速度
- **tokens/s**：模型计算吞吐量（应为常数）
- **gpu_inside [X%]**：GPU 实际计算时间占比（应 >90%）
- **util≈X%**：nvidia-smi 采样的 GPU 利用率

## 注意事项

1. **Batch size 与显存**：
   - batch=16 需要约 25-30GB 显存/进程
   - 3 进程/GPU 需要 75-90GB，80GB GPU 可能 OOM
   - 建议：单 GPU 用 batch=16，多进程用 batch=8

2. **H100 vs A800**：
   - H100 利用率更高（94.5% vs 47.7%）
   - A800 可能是内存带宽限制
   - 优先使用 H100 处理大量任务

3. **小批次问题**：
   - 如果 PDF 只有 1-3 页，实际 batch 大小会很小
   - 这会导致 GPU 利用率下降
   - 无法避免，但整体影响不大

## 相关文件

- `src/numeric_extractor.py`: 批处理主逻辑
- `src/chandra_ocr_tester.py`: `_run_page_ocr_batch()` 方法
- `run_3stream_gpu0.sh`, `run_3stream_gpu1.sh`: 多 GPU 配置
