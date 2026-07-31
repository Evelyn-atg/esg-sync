---
name: h100-a800-performance-comparison
description: H100 与 A800 GPU 性能对比，H100 利用率更高，A800 受内存带宽限制
metadata: 
  node_type: memory
  type: reference
  related: "[[gpu-batching-optimization]], [[chandra-gpu-cluster-setup]]"
  originSessionId: 22294c7b-2e97-4645-a148-1a37910e37ce
  modified: 2026-07-31T09:11:53.036Z
---

# H100 vs A800 性能对比

## 测试环境

- **集群**：Sugon HPC
- **GPU**：2× NVIDIA H100 80GB HBM3 + 2× NVIDIA A800 80GB
- **驱动**：550.54.15 (H100), 545.23.06 (A800)
- **CUDA**：12.4 (H100), 12.3 (A800)
- **模型**：Chandra OCR (9.9GB)
- **批处理**：batch_size=16
- **进程数**：3 streams per GPU

## 性能数据

### H100 (Job 1014)

```
Total pages: 55,538
Total tokens: 123,784,718
Total time: 763,342s
Avg pages/s: 0.07
Avg tokens/s: 162
Avg GPU util: 94.5%
Avg CPU overhead: 4.0%
```

**关键指标：**
- GPU 利用率：94.5%（非常高）
- 处理速度：0.07 pages/s/GPU
- Token 吞吐：162 tok/s
- CPU 开销：4%（合理）

### A800 (Job 1017)

```
Total pages: 17,412
Total tokens: 41,338,056
Total time: 254,260s
Avg pages/s: 0.06
Avg tokens/s: 162
Avg GPU util: 47.7%
Avg CPU overhead: 0%
```

**关键指标：**
- GPU 利用率：47.7%（偏低）
- 处理速度：0.06 pages/s/GPU
- Token 吞吐：162 tok/s（与 H100 相同）
- CPU 开销：0%

## 对比分析

### 相同点

1. **Token 吞吐量一致**：162 tok/s
   - 说明模型计算能力相同
   - GPU 计算单元性能相近

2. **处理速度接近**：0.07 vs 0.06 pages/s
   - 实际差异约 15%

### 关键差异

1. **GPU 利用率**：94.5% vs 47.7%
   - H100 利用率高出 2 倍
   - A800 有一半时间 GPU 空闲

2. **CPU 开销**：4% vs 0%
   - H100 有轻微 CPU 瓶颈
   - A800 CPU 开销可忽略

## 问题诊断

### A800 GPU 利用率低的原因

1. **内存带宽限制**
   - A800 是 A100 的中国特供版
   - NVLink 带宽被削减（400GB/s vs 600GB/s）
   - 大 batch 处理时成为瓶颈

2. **批次大小问题**
   - 某些 PDF 只有 1-3 页
   - 小批次导致 GPU 空闲
   - 无法通过配置解决

3. **CPU/IO 瓶颈**
   - PDF 解析和图片提取
   - 但 CPU overhead 显示为 0%
   - 不太可能是主要原因

4. **特殊 PDF 影响**
   - `2023042100201.pdf`：1 页花 12 分钟
   - 高分辨率图像（2598×3484）
   - 复杂编码（JPEG + JPX）
   - 拖慢整体速度

### 验证测试

```bash
# 检查 A800 性能状态
srun --jobid=1017 nvidia-smi -q -d PERFORMANCE

# 结果：
Performance State: P0（最高性能）
Clocks Event Reasons: 全部 Not Active
# 说明无降频，无热节流
```

## 实际影响

### 9637 份 PDF 任务（2026-07-30）

**配置：**
- H100: list_03, list_04, list_05 (4,818 PDFs)
- A800: list_00, list_01, list_02 (4,818 PDFs)

**进度（2026-07-31）：**
- H100: 59.5% 完成 (2,866/4,818)
- A800: 17.6% 完成 (848/4,818)

**预计完成时间：**
- H100: ~26 小时（明天完成）
- A800: ~88 小时（3.6 天后完成）

**问题：**
- H100 会提前完成并闲置
- A800 需要更长时间
- 总体效率不均衡

## 优化建议

### 1. 优先使用 H100

- 处理大量任务时优先分配 H100
- A800 作为补充或处理小批量任务

### 2. 动态批次调整

```python
# 检测 GPU 利用率
if gpu_util < 80%:
    batch_size = min(batch_size * 2, 32)
elif gpu_util > 95%:
    batch_size = max(batch_size // 2, 8)
```

### 3. 工作负载均衡

```bash
# 统计每个 list 的表格页数
for lst in list_*; do
    count=$(grep "Detected.*table pages" logs/ocr_${lst}_job*.log | wc -l)
    echo "$lst: $count pages"
done

# 均匀分配到不同 GPU
```

### 4. 特殊 PDF 处理

```bash
# 检测异常 PDF（处理时间 > 5 分钟/页）
grep "OCR SUMMARY" logs/*.log | awk -F'|' '{
    split($1, a, " ");
    split(a[6], b, "s");
    pages = split($2, c, "/");
    if (b[1] / pages > 300) print $0
}'
```

## 监控命令

### 实时监控 GPU 状态

```bash
# H100
srun --jobid=1014 nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv

# A800
srun --jobid=1017 nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv
```

### 检查性能状态

```bash
srun --jobid=1017 nvidia-smi -q -d PERFORMANCE
```

### 查看任务进度

```bash
# H100 进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job1014.log | awk -F: '{sum+=$2} END {print "H100:", sum, "PDFs"}'

# A800 进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job1017.log | awk -F: '{sum+=$2} END {print "A800:", sum, "PDFs"}'
```

## 结论

- **H100**：性能优秀，GPU 利用率 94.5%，适合大批量任务
- **A800**：性能略低，GPU 利用率 47.7%，受内存带宽限制
- **Token 吞吐量相同**：说明模型计算能力相近
- **建议**：优先使用 H100，A800 作为补充

## 相关文件

- `run_3stream_gpu0.sh`, `run_3stream_gpu1.sh`: GPU 配置文件
- `logs/ocr_list_*_job*.log`: 任务日志
- `compare_gpu_speed.sh`: 性能对比脚本
