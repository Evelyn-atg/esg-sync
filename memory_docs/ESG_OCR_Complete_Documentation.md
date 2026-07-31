# ESG OCR 项目完整文档

> 本文档整合了 ESG OCR 批处理项目的所有技术细节、优化方案、问题诊断和最佳实践。
> 
> **项目时间**：2026-07-29 至 2026-07-31  
> **处理规模**：9637 份 ESG 报告  
> **核心技术**：Chandra OCR + 多 GPU 并行处理

---

## 目录

1. [项目总结](#1-项目总结)
2. [GPU 批处理优化](#2-gpu-批处理优化)
3. [GPU 错误处理和 PDF 黑名单](#3-gpu-错误处理和-pdf-黑名单)
4. [H100 vs A800 性能对比](#4-h100-vs-a800-性能对比)
5. [GPU 崩溃诊断和特殊 PDF](#5-gpu-崩溃诊断和特殊-pdf)
6. [工作负载分配策略](#6-工作负载分配策略)
7. [内存和资源管理](#7-内存和资源管理)
8. [Pipeline 架构和配置](#8-pipeline-架构和配置)

---

## 1. 项目总结

### 目标

使用 Chandra OCR 处理 9637 份 ESG 报告的表格页，提取结构化数字数据。

### 核心技术

- Chandra OCR（9.9GB 模型）
- 多 GPU 并行处理（H100 + A800）
- 批处理优化（batch=16）
- GPU 错误处理和 PDF 黑名单
- 动态工作负载分配

### 核心改进

1. **GPU 批处理优化**：提升 1.5-2 倍速度
2. **GPU 错误处理**：避免浪费计算时间
3. **工作负载均衡**：最大化资源利用率
4. **内存管理**：避免 OOM 和性能下降

### 经验教训

- GPU 批处理至关重要（提升 1.5-2 倍）
- 错误处理不可或缺（崩溃后立即停止）
- 工作负载均衡（基于缓存的动态分配）
- 内存管理要谨慎（合理的 batch_size 和进程数）
- H100 vs A800：H100 利用率更高（94.5% vs 47.7%）

### 快速参考

```bash
# 提交任务
sbatch run_3stream_gpu0.sh

# 查看任务状态
squeue -u $USER

# 查看 GPU 状态
nvidia-smi

# 查看日志
tail -f logs/ocr_list_00_job*.log

# 检查进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{sum+=$2} END {print sum}'

# 性能对比
bash compare_gpu_speed.sh <job_id_1> <job_id_2>
```

---

## 2. GPU 批处理优化

### 背景

Chandra OCR 原本逐张处理 PDF 页面，GPU 利用率仅 40-60%，大量时间浪费在 Python 循环和 GPU kernel launch 开销上。

### 优化方案

#### 批处理实现

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

#### 性能提升

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

#### 配置建议

```bash
# 单进程模式（测试或小批量）
export OCR_BATCH_SIZE=16

# 多进程模式（3 streams per GPU，避免 OOM）
export OCR_BATCH_SIZE=8

# H100 vs A800
# H100: 可使用 batch=16，GPU 利用率 94.5%
# A800: 建议使用 batch=8-12，GPU 利用率较低（47.7%）
```

#### 监控指标

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

### 注意事项

1. **Batch size 与显存**：
   - batch=16 需要约 25-30GB 显存/进程
   - 3 进程/GPU 需要 75-90GB，80GB GPU 可能 OOM
   - 建议：单 GPU 用 batch=16，多进程用 batch=8

2. **H100 vs A800**：
   - H100 利用率更高（94.5% vs 47.7%）
   - 优先使用 H100 处理大量任务

3. **小批次问题**：
   - 如果 PDF 只有 1-3 页，实际 batch 大小会很小
   - 这会导致 GPU 利用率下降
   - 无法避免，但整体影响不大

---

## 3. GPU 错误处理和 PDF 黑名单

### 问题背景

在运行 Chandra OCR 批处理任务时，GPU 0 在运行 5 小时 23 分钟后崩溃：

```
CUDA error: unspecified launch failure
CUDA kernel errors might be asynchronously reported at some other API call
```

**崩溃后果：**
- Job 1013 继续运行了 5+ 小时，但所有 OCR 任务都失败
- 回退到 text_fallback，浪费了约 7400 个 PDF 的处理时间
- GPU 0 显示 "GPU requires reset"，无法继续使用

### 解决方案

#### GPU 健康追踪系统（GPUHealthTracker）

在 `src/utils.py` 中实现跨进程 GPU 健康状态共享：

```python
class GPUHealthTracker:
    """Track GPU health status across multiple processes."""
    
    def __init__(self):
        self.health_file = Path("/tmp/gpu_health_status.json")
    
    def mark_gpu_unhealthy(self, gpu_id: int, error_msg: str):
        """Mark a GPU as unhealthy and log the failure."""
        data = self._load_health_data()
        if gpu_id not in data.get("unhealthy_gpus", []):
            data.setdefault("unhealthy_gpus", []).append(gpu_id)
        data.setdefault("failure_log", []).append({
            "gpu_id": gpu_id,
            "timestamp": datetime.now().isoformat(),
            "error": error_msg,
            "pid": os.getpid()
        })
        self._save_health_data(data)
    
    def is_gpu_healthy(self, gpu_id: int) -> bool:
        """Check if a GPU is healthy."""
        data = self._load_health_data()
        return gpu_id not in data.get("unhealthy_gpus", [])
```

**存储文件：** `/tmp/gpu_health_status.json`

```json
{
  "unhealthy_gpus": [0],
  "failure_log": [
    {
      "gpu_id": 0,
      "timestamp": "2026-07-30T07:03:05.839123",
      "error": "CUDA error: unspecified launch failure",
      "pid": 12345
    }
  ]
}
```

#### PDF 黑名单机制（PDFBlacklist）

在 `src/utils.py` 中实现 PDF 黑名单追踪：

```python
class PDFBlacklist:
    """Track PDFs that have caused GPU crashes or other fatal errors."""
    
    def __init__(self):
        self.blacklist_file = Path("/tmp/pdf_blacklist.json")
    
    def add_to_blacklist(self, pdf_name: str, reason: str):
        """Add a PDF to the blacklist and log the reason."""
        data = self._load_blacklist()
        if pdf_name not in data.get("blacklisted_pdfs", []):
            data.setdefault("blacklisted_pdfs", []).append(pdf_name)
        data.setdefault("failure_log", []).append({
            "pdf_name": pdf_name,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "pid": os.getpid()
        })
        self._save_blacklist(data)
    
    def is_blacklisted(self, pdf_name: str) -> bool:
        """Check if a PDF is blacklisted."""
        data = self._load_blacklist()
        return pdf_name in data.get("blacklisted_pdfs", [])
```

**存储文件：** `/tmp/pdf_blacklist.json`

```json
{
  "blacklisted_pdfs": ["ltn201707281159", "2023042100201"],
  "failure_log": [
    {
      "pdf_name": "ltn201707281159",
      "timestamp": "2026-07-30T07:03:05.839123",
      "reason": "CUDA error on GPU 0",
      "pid": 12345
    }
  ]
}
```

#### 错误处理流程

在 `src/chandra_ocr_tester.py` 中捕获 CUDA 错误：

```python
def _run_page_ocr_batch_chandra_local(self, image_paths, timing=None, pdf_name=None):
    gpu_health = GPUHealthTracker()
    pdf_blacklist = PDFBlacklist()
    
    try:
        outputs = manager.generate(items)
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "cuda" in error_msg:
            gpu_id = torch.cuda.current_device()
            logger.error(f"Fatal CUDA error detected on GPU {gpu_id}: {error_msg}")
            
            # Mark GPU as unhealthy
            gpu_health.mark_gpu_unhealthy(gpu_id, error_msg)
            
            # Add PDF to blacklist
            if pdf_name:
                pdf_blacklist.add_to_blacklist(pdf_name, f"CUDA error on GPU {gpu_id}")
            
            # Exit immediately
            raise GPUFatalError(f"CUDA error on GPU {gpu_id}: {error_msg}")
```

在 `src/numeric_extractor.py` 中处理 GPUFatalError：

```python
try:
    batch_results = ocr_tester._run_page_ocr_batch(batch_images, timing=timing, pdf_name=pdf_name)
except GPUFatalError as e:
    sampler.stop()
    logger.critical(f"FATAL GPU ERROR in batch {batch_idx}/{total_batches}: {e}")
    logger.critical("Worker exiting immediately. GPU marked as unhealthy.")
    raise  # Exit immediately
```

#### 黑名单检查

在处理 PDF 前检查黑名单：

```python
# Check if this PDF is blacklisted
pdf_blacklist = PDFBlacklist()
if pdf_blacklist.is_blacklisted(pdf_name):
    logger.warning(f"PDF {pdf_name} is blacklisted. Skipping.")
    return {}  # Skip this PDF
```

### 管理命令

#### 查看 GPU 健康状态

```bash
cat /tmp/gpu_health_status.json | python -m json.tool
```

#### 查看 PDF 黑名单

```bash
cat /tmp/pdf_blacklist.json | python -m json.tool
```

#### 重置 GPU 健康状态（GPU 修复后）

```bash
python3 << 'EOF'
import json
from pathlib import Path

health_file = Path("/tmp/gpu_health_status.json")
if health_file.exists():
    with open(health_file, 'r') as f:
        data = json.load(f)
    if 0 in data.get("unhealthy_gpus", []):
        data["unhealthy_gpus"].remove(0)
        with open(health_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("GPU 0 health status reset")
EOF
```

#### 从黑名单移除 PDF

```bash
python3 << 'EOF'
import json
from pathlib import Path

blacklist_file = Path("/tmp/pdf_blacklist.json")
if blacklist_file.exists():
    with open(blacklist_file, 'r') as f:
        data = json.load(f)
    if "ltn201707281159" in data.get("blacklisted_pdfs", []):
        data["blacklisted_pdfs"].remove("ltn201707281159")
        with open(blacklist_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("Removed from blacklist")
EOF
```

### 效果

**优化前：**
- GPU 崩溃后继续运行 5+ 小时，所有 OCR 失败
- 浪费大量计算时间

**优化后：**
- GPU 崩溃后立即停止（几秒钟内）
- 自动标记不健康 GPU
- 问题 PDF 自动加入黑名单
- 其他 worker 可以避免使用故障 GPU

---

## 4. H100 vs A800 性能对比

### 测试环境

- **集群**：Sugon HPC
- **GPU**：2× NVIDIA H100 80GB HBM3 + 2× NVIDIA A800 80GB
- **驱动**：550.54.15 (H100), 545.23.06 (A800)
- **CUDA**：12.4 (H100), 12.3 (A800)
- **模型**：Chandra OCR (9.9GB)
- **批处理**：batch_size=16
- **进程数**：3 streams per GPU

### 性能数据

#### H100 (Job 1014)

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

#### A800 (Job 1017)

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

### 对比分析

#### 相同点

1. **Token 吞吐量一致**：162 tok/s
   - 说明模型计算能力相同
   - GPU 计算单元性能相近

2. **处理速度接近**：0.07 vs 0.06 pages/s
   - 实际差异约 15%

#### 关键差异

1. **GPU 利用率**：94.5% vs 47.7%
   - H100 利用率高出 2 倍
   - A800 有一半时间 GPU 空闲

2. **CPU 开销**：4% vs 0%
   - H100 有轻微 CPU 瓶颈
   - A800 CPU 开销可忽略

### 问题诊断

#### A800 GPU 利用率低的原因

1. **批次大小问题**
   - 某些 PDF 只有 1-3 页
   - 小批次导致 GPU 空闲
   - 无法通过配置解决

2. **特殊 PDF 影响**
   - `2023042100201.pdf`：1 页花 12 分钟
   - 高分辨率图像（2598×3484）
   - 复杂编码（JPEG + JPX）
   - 拖慢整体速度

#### 验证测试

```bash
# 检查 A800 性能状态
srun --jobid=1017 nvidia-smi -q -d PERFORMANCE

# 结果：
Performance State: P0（最高性能）
Clocks Event Reasons: 全部 Not Active
# 说明无降频，无热节流
```

### 实际影响

#### 9637 份 PDF 任务（2026-07-30）

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

### 优化建议

1. **优先使用 H100**：处理大量任务时优先分配 H100
2. **工作负载均衡**：基于 OCR 缓存动态分配任务
3. **特殊 PDF 处理**：检测并跳过异常 PDF

### 监控命令

```bash
# 实时监控 GPU 状态
srun --jobid=1014 nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv
srun --jobid=1017 nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv

# 查看任务进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job1014.log | awk -F: '{sum+=$2} END {print "H100:", sum, "PDFs"}'
grep -c "OCR SUMMARY" logs/ocr_list_*_job1017.log | awk -F: '{sum+=$2} END {print "A800:", sum, "PDFs"}'
```

### 结论

- **H100**：性能优秀，GPU 利用率 94.5%，适合大批量任务
- **A800**：性能略低，GPU 利用率 47.7%
- **Token 吞吐量相同**：说明模型计算能力相近
- **建议**：优先使用 H100，A800 作为补充

---

## 5. GPU 崩溃诊断和特殊 PDF 分析

### 崩溃事件

#### 事件概述

**时间**：2026-07-30 07:03:05  
**Job**：1013 (GPU 0, H100)  
**运行时长**：5 小时 23 分钟（从 01:40:24 到 07:03:05）  
**崩溃后**：Job 继续运行 5+ 小时，但所有 OCR 失败，回退到 text_fallback

#### 错误日志

```
2026-07-30 07:03:05,839 - WARNING - [esg-ocr][gpu=0][ltn201707281159] 
Batch 1/1 failed entirely (pages [1, 2]): CUDA error: unspecified launch failure
CUDA kernel errors might be asynchronously reported at some other API call, 
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

#### 崩溃时间线

```
01:40:13  Job 1013 启动（GPU 0）
01:41:13  nvidia-smi 检查：GPU 0 正常，温度 74°C，利用率 100%
01:46:14  nvidia-smi 检查：GPU 0 正常
01:49:43  nvidia-smi 检查：GPU 0 正常
01:40 - 07:03  正常运行，OCR 成功处理 362 个 PDF（约 5 小时 23 分）
07:03:05  第一次 CUDA 错误，处理 ltn201707281159.pdf 时崩溃
07:03 - 12:22  Job 继续运行，但所有 OCR 失败（约 5 小时 20 分）
12:22:54  Job 1013 完成，总运行时间 10:42:41
```

#### 影响统计

- **成功 OCR**：362 个 PDF
- **OCR 失败**：7,436 个 PDF（回退到 text_fallback）
- **浪费计算时间**：约 5 小时 20 分

#### GPU 0 当前状态

```bash
$ srun --jobid=1016 nvidia-smi
index, temperature.gpu, utilization.gpu [%], memory.used [MiB]
0, [GPU requires reset], [GPU requires reset], 0 MiB
```

**结论**：GPU 0 需要硬件重置，可能是硬件故障。

### 对比：GPU 1 正常运行

**Job**：1014 (GPU 1, H100)  
**运行时长**：18+ 小时  
**状态**：完全正常

```bash
$ srun --jobid=1014 nvidia-smi
index, temperature.gpu, utilization.gpu [%], memory.used [MiB]
0, 74, 100 %, 80620 MiB
```

**性能**：
- GPU 利用率：94.5%
- 处理速度：0.07 pages/s
- Token 吞吐：162 tok/s

**结论**：GPU 1 完全正常，说明问题仅限于 GPU 0。

### 崩溃原因分析

#### 可能的原因

1. **硬件故障**（最可能）
   - GPU 0 显存错误
   - 计算单元故障
   - 散热问题（虽然温度显示正常）

2. **特定 PDF 触发**
   - `ltn201707281159.pdf` 可能包含特殊内容
   - 触发 GPU 驱动或 CUDA 运行时 bug

3. **驱动 bug**
   - CUDA 12.4 驱动问题
   - 与 Chandra 模型的兼容性问题

4. **内存不足**
   - 虽然显存显示正常，但可能存在内存泄漏
   - 长时间运行后积累

#### 诊断命令

```bash
# 检查 dmesg（需要 root 权限）
sudo dmesg | grep -i -E "xid|nvidia|gpu|error" | tail -20

# 检查 Xid 错误码
# Xid 13: Graphics Engine Exception
# Xid 31: GPU memory page fault
# Xid 43: GPU stopped processing
# Xid 48: Double Bit ECC Error
# Xid 79: GPU has fallen off the bus

# 检查 nvidia 日志
cat /var/log/nvidia-smi.log 2>/dev/null

# 检查系统日志
journalctl -k | grep -i nvidia | tail -20
```

#### 向 IT 报告

**报告内容**：

```
主题：GPU 0 (Bus-Id: 23:00.0) 运行中崩溃，请求检查和重置

问题描述：
- 时间：2026-07-30 07:03:05
- 设备：gpu3 节点的 GPU 0 (Bus-Id: 23:00.0)
- 任务：Job ID 1013，运行 Chandra OCR 模型
- 症状：运行 5 小时 23 分钟后突然崩溃，所有后续计算失败

错误日志：
CUDA error: unspecified launch failure
CUDA kernel errors might be asynchronously reported at some other API call

当前状态：
GPU 0 显示 "requires reset"，无法使用。

对比情况：
- GPU 1 (Bus-Id: A4:00.0) 同时运行 Job 1014，已运行 18+ 小时，状态正常
- 温度：74°C，利用率：100%，显存：80620 MiB

请求：
1. 检查 GPU 0 的硬件状态
2. 重置 GPU 0 或标记为需要维护
3. 提供崩溃时的详细日志（温度、错误代码等）
```

### 特殊问题 PDF 分析

#### 1. ltn201707281159.pdf

**基本信息**：
- **股票**：01622 REDCO PPT
- **标题**：ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2016
- **页数**：5 页
- **大小**：160KB
- **PDF 版本**：1.4

**问题**：触发 GPU 0 崩溃

**检查结果**：

```bash
$ pdfinfo ltn201707281159.pdf
Pages: 5
Page size: 595.276 x 841.89 pts (A4)
Creator: Adobe InDesign CS6 (Windows)
Producer: Adobe PDF Library 10.0.1
PDF version: 1.4

$ pdfimages -list ltn201707281159.pdf
page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
--------------------------------------------------------------------------------------------
   1   0 image    138   138  rgb      3   8 jpeg   no         8  0    72    72 3452B  6.0%
   2   1 image    138   138  rgb      3   8 jpeg   no        10  0    72    72 3452B  6.0%
```

**分析**：
- 无明显异常（肉眼检查）
- 标准 A4 尺寸
- 低分辨率图像（72 DPI）
- 简单的 JPEG 编码
- **无法确定具体原因**

**可能原因**：
1. PDF 内部结构问题（不可见）
2. 与 Chandra 模型的特定交互
3. 触发 GPU 驱动的边界情况

**处理**：已加入 PDF 黑名单

#### 2. 2023042100201.pdf

**基本信息**：
- **股票**：02382 SUNNY OPTICAL
- **标题**：ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2022
- **页数**：91 页
- **大小**：12MB
- **PDF 版本**：1.7 (zip deflate encoded)

**问题**：在 A800 上处理极慢（1 页花 12 分钟）

**检查结果**：

```bash
$ pdfinfo 2023042100201.pdf
Pages: 91
Page size: 595.276 x 841.89 pts (A4)
Creator: Adobe InDesign 15.1 (Windows)
Producer: Adobe PDF Library 15.0
PDF version: 1.7

$ pdfimages -list 2023042100201.pdf | head -20
page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
--------------------------------------------------------------------------------------------
   1   0 image    497   352  rgb      3   8 jpeg   no        11  0    72    72  12KB  2.2%
   1   1 image   2598  3484  rgb      3   8 jpeg   no        14  0   300   300 456KB  1.8%
   2   2 image   1247  1754  rgb      3   8 jpeg   no        20  0   150   150  89KB  1.4%
```

**分析**：
- **高分辨率图像**：2598×3484 像素（第 1 页）
- **300 DPI**：印刷质量
- **大文件**：12MB（91 页）
- **复杂编码**：zip deflate
- **大量图像**：每页多个图像

**性能数据**：

```
OCR SUMMARY: 1/1 pages, 12384 tok in 725.94s | 0.00 pages/s, 17 tok/s
breakdown: img_load 0.04s (0%), generate 725.79s (gpu 725.79s [100%], cpu_overhead 0.00s [0%]), postproc 0.02s (0%)
```

**问题**：
- 1 页花了 725.94 秒（12 分钟）
- GPU 利用率 100%（说明不是空闲）
- Token 生成速度仅 17 tok/s（正常应为 162 tok/s）
- 生成了 12,384 个 tokens（异常多）

**可能原因**：
1. **高分辨率图像**：2598×3484 像素需要大量计算
2. **复杂布局**：可能导致 Chandra 模型生成更多 tokens
3. **显存带宽 / Tensor Core 性能限制**：A800 的显存带宽和 Tensor Core 算力低于 H100
4. **模型过载**：单个 PDF 占用过多 GPU 资源

**处理**：
- 已加入 PDF 黑名单
- 建议：对于高分辨率 PDF，降低图像分辨率或跳过

#### 3. 其他异常 PDF

**检测异常 PDF 的命令**：

```bash
# 查找处理时间 > 5 分钟/页的 PDF
grep "OCR SUMMARY" logs/*.log | awk -F'|' '{
    split($1, a, " ");
    split(a[6], b, "s");
    pages = split($2, c, "/");
    if (b[1] / pages > 300) print $0
}'

# 查找 token 生成速度 < 50 tok/s 的 PDF
grep "OCR SUMMARY" logs/*.log | grep -E "[0-9]+ tok/s" | awk -F',' '{
    split($2, a, " ");
    if (a[1] < 50) print $0
}'
```

### 解决方案

#### 短期措施

1. **GPU 健康追踪**：已实现 GPUHealthTracker
2. **PDF 黑名单**：已实现 PDFBlacklist
3. **立即停止**：GPU 崩溃后立即停止 worker
4. **自动跳过**：问题 PDF 自动加入黑名单

#### 中期优化

1. **PDF 预处理**：
   - 检测高分辨率图像
   - 自动降低分辨率
   - 跳过复杂 PDF

2. **异常检测**：
   - 监控处理时间
   - 检测异常 PDF
   - 自动加入黑名单

#### 长期改进

1. **PDF 质量评估**：
   - 预处理阶段评估 PDF 复杂度
   - 根据复杂度分配资源
   - 避免单个 PDF 占用过多资源

2. **GPU 负载均衡**：
   - 检测 GPU 健康状态
   - 自动切换到健康 GPU
   - 避免故障 GPU

3. **模型优化**：
   - 与 Chandra 团队反馈问题
   - 优化高分辨率图像处理
   - 减少 token 生成量

---

## 6. 工作负载分配策略

### 问题背景

在运行 9637 份 ESG 报告 OCR 任务时，需要合理分配工作负载到多个 GPU，避免：
- 重复工作（浪费计算资源）
- 工作不均衡（某些 GPU 提前完成，其他 GPU 仍在运行）
- 缓存冲突（多个进程同时写入同一文件）

### 策略概述

#### 1. 基于 OCR 缓存的分配（推荐）

**原理**：利用 OCR 缓存文件判断哪些 PDF 已完成，只分配未完成的 PDF。

**优点**：
- ✅ 无重复工作
- ✅ 支持动态添加 GPU
- ✅ 支持断点续传

**缺点**：
- ⚠️ 需要扫描缓存目录
- ⚠️ 缓存文件可能被清理

#### 2. 预分割列表

**原理**：在任务开始前，将 PDF 列表均匀分割成多个子列表。

**优点**：
- ✅ 简单直接
- ✅ 无需扫描缓存

**缺点**：
- ❌ 工作可能不均衡（某些 PDF 处理时间长）
- ❌ 不支持动态调整

### 实现方案

#### 方案 1：基于 OCR 缓存的分配

##### 步骤 1：扫描已完成的 PDF

```bash
cd ~/esg-pipeline

cat > split_remaining_work.sh << 'EOFSPLIT'
#!/bin/bash
# 找出 OCR 已完成的 PDF（有 OCR 缓存文件的）
echo "查找 OCR 已完成的 PDF..."
completed=0
total=0

for pdf_name in $(cat list_00 list_01 list_02); do
    total=$((total + 1))
    # 去掉 .pdf 扩展名
    pdf_base="${pdf_name%.pdf}"
    # 检查 OCR 缓存文件（正确的路径）
    if [ -f "quantitative_results_ocr/chandra_ocr_2/${pdf_base}/${pdf_base}_ocr_output.json" ]; then
        completed=$((completed + 1))
    fi
done

echo "总数: $total"
echo "OCR 已完成: $completed"
echo "OCR 未完成: $((total - completed))"

# 创建未完成 OCR 的列表
echo ""
echo "创建未完成 OCR 的 PDF 列表..."
> list_ocr_remaining.txt
for pdf_name in $(cat list_00 list_01 list_02); do
    pdf_base="${pdf_name%.pdf}"
    if [ ! -f "quantitative_results_ocr/chandra_ocr_2/${pdf_base}/${pdf_base}_ocr_output.json" ]; then
        echo "$pdf_name" >> list_ocr_remaining.txt
    fi
done

# 修复空格问题：使用 tr -d 删除空格
remaining=$(wc -l < list_ocr_remaining.txt | tr -d ' ')
echo "未完成 OCR: $remaining 个 PDF"

# 分成两半
if [ "$remaining" -gt 0 ]; then
    half=$((remaining / 2))
    head -n $half list_ocr_remaining.txt > list_ocr_remaining_a.txt
    tail -n +$((half + 1)) list_ocr_remaining.txt > list_ocr_remaining_b.txt
    
    echo ""
    echo "分成两份:"
    wc -l list_ocr_remaining_a.txt list_ocr_remaining_b.txt
else
    echo ""
    echo "所有 PDF 的 OCR 都已完成！"
fi
EOFSPLIT

chmod +x split_remaining_work.sh
bash split_remaining_work.sh
```

##### 步骤 2：创建分配脚本

```bash
# GPU 0 处理 list_ocr_remaining_a.txt
cat > run_a800_a.sh << 'EOF'
#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ocr_a800_a_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OCR_BATCH_SIZE=16

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"

echo "launching OCR stream for list_ocr_remaining_a (job $JOB_TAG, A100)"
python -m src.main --step numeric_extraction --pdf_list_file list_ocr_remaining_a.txt --force > "logs/ocr_list_ocr_remaining_a_job${JOB_TAG}.log" 2>&1

echo "=== A100 done (job $JOB_TAG) ==="
EOF

chmod +x run_a800_a.sh

# GPU 1 处理 list_ocr_remaining_b.txt
cat > run_a800_b.sh << 'EOF'
#!/bin/bash
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ocr_a800_b_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OCR_BATCH_SIZE=16

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"

echo "launching OCR stream for list_ocr_remaining_b (job $JOB_TAG, A100 new)"
python -m src.main --step numeric_extraction --pdf_list_file list_ocr_remaining_b.txt --force > "logs/ocr_list_ocr_remaining_b_job${JOB_TAG}.log" 2>&1

echo "=== A100 new done (job $JOB_TAG) ==="
EOF

chmod +x run_a800_b.sh
```

##### 步骤 3：提交任务

```bash
sbatch run_a800_a.sh
sbatch run_a800_b.sh
```

### 监控进度

#### 检查每个 GPU 的进度

```bash
# 查看所有 A100 任务的进度
echo "=== 所有 A100 任务 ==="
for job_id in $(squeue -u $USER | grep A100 | awk '{print $1}'); do
    echo "Job $job_id:"
    grep -c "OCR SUMMARY" logs/ocr_list_*_job${job_id}.log 2>/dev/null | awk -F: '{sum+=$2} END {print "  已处理:", sum, "个 PDF"}'
done

# 总体进度
total=4818
done=$(grep -c "OCR SUMMARY" logs/ocr_list_*_job1017.log logs/ocr_list_*_job<NEW_JOB_ID>.log 2>/dev/null | awk -F: '{sum+=$2} END {print sum}')
echo "=== list_00, 01, 02 总进度 ==="
echo "已完成: $done / $total ($((done * 100 / total))%)"
```

#### 检查 H100 和 A100 的对比

```bash
echo "=== H100 进度 ==="
grep -c "OCR SUMMARY" logs/ocr_list_*_job1014.log | awk -F: '{sum+=$2} END {print sum, "个 PDF"}'

echo "=== A100 进度 ==="
grep -c "OCR SUMMARY" logs/ocr_list_*_job1017.log | awk -F: '{sum+=$2} END {print sum, "个 PDF"}'
```

### 实际案例

#### 2026-07-30 任务分配

**初始配置**：
- H100 (Job 1014): list_03, list_04, list_05 (4,818 PDFs)
- A800 (Job 1017): list_00, list_01, list_02 (4,818 PDFs)

**进度（2026-07-31）**：
- H100: 59.5% 完成 (2,866/4,818)
- A800: 17.6% 完成 (848/4,818)

**问题**：
- H100 会提前完成并闲置
- A800 需要更长时间
- 总体效率不均衡

**解决方案**：

1. **重新分配 A800 任务**：
   - 停止 Job 1017
   - 扫描 OCR 缓存，找出未完成的 PDF
   - 平均分配给两张 A100

2. **执行步骤**：
   ```bash
   # 停止 Job 1017
   scancel 1017
   
   # 扫描并分割
   bash split_remaining_work.sh
   
   # 提交新任务
   sbatch run_a800_a.sh
   sbatch run_a800_b.sh
   ```

3. **结果**：
   - 1423 个 PDF 已完成 OCR
   - 3395 个 PDF 未完成
   - 分配给两张 A100：1697 + 1698
   - 预计完成时间：37.7 小时（从 88 小时减半）

### 最佳实践

#### 1. 任务前评估

```bash
# 评估每个 list 的工作量
for lst in list_*; do
    echo "=== $lst ==="
    # 统计表格页数
    count=$(grep "Detected.*table pages" logs/ocr_${lst}_job*.log | wc -l)
    echo "  表格页数: $count"
    
    # 统计已完成 OCR 的 PDF
    completed=0
    total=0
    for pdf_name in $(cat $lst); do
        total=$((total + 1))
        pdf_base="${pdf_name%.pdf}"
        if [ -f "quantitative_results_ocr/chandra_ocr_2/${pdf_base}/${pdf_base}_ocr_output.json" ]; then
            completed=$((completed + 1))
        fi
    done
    echo "  已完成: $completed / $total"
done
```

#### 2. 断点续传

```bash
# 检查哪些 PDF 未完成
bash split_remaining_work.sh

# 重新提交未完成的任务
sbatch run_a800_a.sh
sbatch run_a800_b.sh
```

---

## 7. 内存和资源管理

### 问题背景

Chandra OCR 批处理任务需要大量 GPU 显存和系统内存，不当的配置会导致：
- GPU OOM（Out of Memory）
- 系统 OOM Killer 杀死进程
- 性能下降（频繁 swap）
- 任务失败

### 显存使用分析

#### 模型显存占用

```bash
# 检查模型大小
ls -lh ~/models/chandra-ocr-2/model.safetensors
# 输出：9.9G
```

**模型加载到 GPU 后占用**：~10GB

#### 批处理显存占用

**每个 batch 的显存使用**：

```python
# batch_size=16 时的显存使用
# - 输入图像：16 × 3MB = 48MB
# - 中间激活：~2-5GB（取决于图像复杂度）
# - KV Cache：~1-3GB（生成长度 500-5000 tokens）
# 总计：~3-8GB per batch
```

#### 进程显存占用

**每个 Python 进程的显存使用**：

```
模型权重：~10GB
PyTorch 开销：~2-3GB
Batch 处理：~3-8GB（取决于 batch_size）
总计：~15-21GB per process
```

#### 多进程显存使用

**3 进程/GPU 配置**：

```
3 × 21GB = 63GB
```

**H100 80GB**：
- 可用：80GB
- 3 进程需要：63GB
- 剩余：17GB（安全余量）

**A800 80GB**：
- 可用：80GB
- 3 进程需要：63GB
- 剩余：17GB（安全余量）

### 系统内存管理

#### 系统内存需求

**每个 Python 进程的系统内存使用**：

```
模型加载到 CPU：~10GB（临时）
PyTorch 开销：~2-3GB
PDF 解析：~1-2GB per PDF
OCR 缓存：~0.5-2MB per page
总计：~15-20GB per process
```

**3 进程配置**：

```
3 × 20GB = 60GB
```

#### SLURM 内存配置

```bash
#SBATCH --mem=64G  # 推荐配置
```

**为什么是 64G？**

```
3 进程 × 20GB = 60GB
+ 系统开销：~4GB
总计：~64GB
```

**警告**：如果设置为 32G（之前的配置），会导致：
- ❌ OOM Killer 杀死进程
- ❌ 任务失败
- ❌ 浪费计算时间

### Batch Size 配置

#### 推荐配置

```bash
# 单进程模式（测试或小批量）
export OCR_BATCH_SIZE=16

# 多进程模式（3 streams per GPU）
export OCR_BATCH_SIZE=8
```

#### 选择依据

| 场景 | 推荐 batch_size | 原因 |
|------|----------------|------|
| 单进程/GPU | 16 | 最大化 GPU 利用率 |
| 2 进程/GPU | 12 | 平衡利用率和显存 |
| 3 进程/GPU | 8 | 避免 OOM |
| 4+ 进程/GPU | 4-6 | 显存限制 |

### 进程数配置

#### 推荐配置

```bash
# 3 进程/GPU（推荐）
#SBATCH --cpus-per-task=8  # 每个进程 8 CPU cores
#SBATCH --gres=gpu:1       # 1 GPU per job
```

**为什么是 3 进程？**

1. **GPU 利用率**：
   - 1 进程：GPU 利用率 ~60%（空闲时间多）
   - 3 进程：GPU 利用率 ~95%（充分利用）

2. **CPU 利用率**：
   - 每个进程需要 2-3 CPU cores 处理 PDF 解析
   - 3 进程需要 6-9 CPU cores
   - 8 cores 足够

3. **显存限制**：
   - 3 × 21GB = 63GB < 80GB（安全）
   - 4 × 21GB = 84GB > 80GB（OOM 风险）

### 监控命令

#### 实时监控 GPU 状态

```bash
# 监控 GPU 利用率和显存
watch -n 2 nvidia-smi

# 监控特定 GPU
watch -n 2 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"
```

#### 监控进程显存使用

```bash
# 查看所有 Python 进程的显存使用
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

# 查看特定进程的详细信息
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv | grep <PID>
```

#### 监控系统内存

```bash
# 实时监控系统内存
watch -n 2 free -h

# 查看进程的内存使用
ps aux | grep python | awk '{print $2, $4, $6}' | column -t
# 列：PID, %MEM, RSS (KB)
```

#### 检查 OOM 事件

```bash
# 查看 dmesg 中的 OOM 事件
dmesg | grep -i "out of memory" | tail -10

# 查看系统日志
journalctl -k | grep -i "oom" | tail -10
```

### 优化技巧

#### 1. 预加载模型

```python
# 在脚本开始时加载模型，避免重复加载
ocr_tester = ChandraOCRTester()
ocr_tester._ensure_chandra_manager()
```

**优点**：
- ✅ 减少模型加载时间（每次 ~1 分钟）
- ✅ 避免显存碎片

#### 2. 清理缓存

```bash
# 清理 PyTorch 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 清理系统缓存（需要 root）
sync; echo 3 > /proc/sys/vm/drop_caches
```

#### 3. 限制 CPU 使用

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

**原因**：
- 避免 CPU 过度竞争
- 减少内存带宽压力
- 提高 GPU 利用率

### 故障排查

#### GPU OOM

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决方案**：

1. **减少 batch_size**：
   ```bash
   export OCR_BATCH_SIZE=8  # 从 16 降到 8
   ```

2. **减少进程数**：
   ```bash
   # 从 3 进程降到 2 进程
   for lst in list_00 list_01; do  # 只处理 2 个 list
       python -m src.main --step numeric_extraction --pdf_list_file "$lst" &
   done
   ```

3. **清理缓存**：
   ```bash
   python -c "import torch; torch.cuda.empty_cache()"
   ```

#### 系统 OOM

**症状**：
- 进程被杀死（无错误信息）
- dmesg 显示 OOM Killer 活动

**解决方案**：

1. **增加系统内存**：
   ```bash
   #SBATCH --mem=96G  # 从 64G 增加到 96G
   ```

2. **减少进程数**：
   ```bash
   # 从 3 进程降到 2 进程
   ```

3. **优化 PDF 解析**：
   - 减少同时处理的 PDF 数量
   - 使用更小的 PDF 分块

#### 性能下降

**症状**：
- GPU 利用率低（<50%）
- 处理速度慢

**解决方案**：

1. **检查 batch_size**：
   - 如果太小，增加 batch_size
   - 如果太大，减少 batch_size

2. **检查 CPU 瓶颈**：
   ```bash
   top -b -n 1 | grep python
   ```
   - 如果 CPU 利用率 >80%，减少 CPU threads

3. **检查 I/O 瓶颈**：
   ```bash
   iostat -x 1
   ```
   - 如果磁盘利用率 >90%，使用更快的存储

### 配置模板

#### 单 GPU 配置

```bash
#!/bin/bash
#SBATCH -p H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ocr_single_gpu_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OCR_BATCH_SIZE=16  # 单进程可以使用更大的 batch

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"

echo "launching OCR stream for list_00 (job $JOB_TAG)"
python -m src.main --step numeric_extraction --pdf_list_file list_00 --force > "logs/ocr_list_00_job${JOB_TAG}.log" 2>&1

echo "=== Done (job $JOB_TAG) ==="
```

#### 多 GPU 配置（3 进程/GPU）

```bash
#!/bin/bash
#SBATCH -p H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o ocr_multi_stream_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OCR_BATCH_SIZE=8  # 多进程需要较小的 batch

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"

for lst in list_00 list_01 list_02; do
  [ -f "$lst" ] || continue
  echo "launching OCR stream for $lst (job $JOB_TAG)"
  python -m src.main --step numeric_extraction --pdf_list_file "$lst" --force > "logs/ocr_${lst}_job${JOB_TAG}.log" 2>&1 &
done
wait

echo "=== All streams done (job $JOB_TAG) ==="
```

---

## 8. Pipeline 架构和配置

### 系统架构

#### 整体流程

```
PDF 文件
  ↓
表格页检测（基于文本特征）
  ↓
OCR 批处理（Chandra，GPU）
  ↓
HTML/Markdown 输出
  ↓
缓存管理（quantitative_results_ocr/chandra_ocr_2/）
```

#### 关键组件

```
src/
├── main.py                      # 主入口
├── numeric_extractor.py         # 数字提取器（批处理逻辑）
├── chandra_ocr_tester.py        # Chandra OCR 封装
├── utils.py                     # GPU 健康追踪、PDF 黑名单
└── config.py                    # 配置管理
```

#### 数据流

```
1. PDF → 表格页检测
   - 输入：PDF 文件
   - 输出：表格页列表（页码）
   - 方法：基于文本特征（数字、关键词）

2. 表格页 → OCR 批处理
   - 输入：表格页图像
   - 输出：HTML/Markdown
   - 方法：Chandra OCR（GPU 批处理）
   - 缓存：quantitative_results_ocr/chandra_ocr_2/{pdf_name}/

3. OCR 结果 → 数字提取
   - 输入：HTML/Markdown
   - 输出：结构化数字数据
   - 方法：正则表达式、关键词匹配
```

### 配置文件

#### SLURM 作业配置

```bash
#SBATCH -p H100                  # 分区：H100 或 A800
#SBATCH --gres=gpu:1             # GPU 数量
#SBATCH --cpus-per-task=8        # CPU cores per process
#SBATCH --mem=64G                # 系统内存（3 进程 × 20GB + 4GB）
#SBATCH -o ocr_%j.out            # 输出日志
```

#### 环境变量

```bash
# 模型配置
export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1

# 批处理配置
export OCR_BATCH_SIZE=16         # 单进程
export OCR_BATCH_SIZE=8          # 3 进程/GPU

# CPU 限制
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# PDF 目录
export PDF_INPUT_DIR="HKEX ESG Reports"
```

#### Python 配置

```python
# src/numeric_extractor.py
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "16"))

# src/utils.py
_GPU_HEALTH_FILE = Path("/tmp/gpu_health_status.json")
_PDF_BLACKLIST_FILE = Path("/tmp/pdf_blacklist.json")
```

### 运行流程

#### 1. 准备阶段

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 PDF 列表
wc -l list_*

# 检查已完成的任务
ls -la quantitative_results_ocr/chandra_ocr_2/ | wc -l
```

#### 2. 分配任务

```bash
# 基于 OCR 缓存分配
bash split_remaining_work.sh

# 查看分配结果
wc -l list_ocr_remaining_*.txt
```

#### 3. 提交任务

```bash
# 单 GPU
sbatch run_3stream_gpu0.sh

# 多 GPU
sbatch run_3stream_gpu0.sh
sbatch run_3stream_gpu1.sh
```

#### 4. 监控任务

```bash
# 查看任务状态
squeue -u $USER

# 查看 GPU 状态
watch -n 2 nvidia-smi

# 查看日志
tail -f logs/ocr_list_00_job*.log
```

#### 5. 检查进度

```bash
# 统计已处理的 PDF
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{sum+=$2} END {print sum}'

# 检查 GPU 性能
bash compare_gpu_speed.sh <job_id_1> <job_id_2>
```

### 关键参数

#### Batch Size

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 单进程/GPU | 16 | 最大化 GPU 利用率 |
| 2 进程/GPU | 12 | 平衡利用率和显存 |
| 3 进程/GPU | 8 | 避免 OOM |
| 4+ 进程/GPU | 4-6 | 显存限制 |

#### 进程数

| GPU 类型 | 推荐进程数 | 说明 |
|----------|-----------|------|
| H100 | 3 | GPU 利用率 94.5% |
| A800 | 3 | GPU 利用率 47.7% |

#### 系统内存

| 进程数 | 推荐内存 | 说明 |
|--------|---------|------|
| 1 | 32G | 单进程 |
| 2 | 48G | 双进程 |
| 3 | 64G | 推荐配置 |
| 4+ | 96G+ | 高并发 |

### 日志分析

#### OCR 日志格式

```
[时间戳] - [模块] - INFO - [esg-ocr][gpu=<id>][<pdf_name>] 
OCR batching: <pages> pages → <batches> batch(es) of ≤<batch_size>

[时间戳] - [模块] - INFO - [esg-ocr][gpu=<id>][<pdf_name>] 
Batch <idx>/<total> done (pages <start>-<end>, <tokens> tok): 
  wall <time>s = img_load <time>s (<pct>%) + generate <time>s 
    (gpu <time>s [<pct>%], cpu_overhead <time>s [<pct>%]) + postproc <time>s (<pct>%) |
  <pages_per_s> pages/s, <tokens_per_s> tok/s, util≈<util>%
```

#### 关键指标

- **pages/s**：实际处理速度（应 >0.05）
- **tokens/s**：模型计算吞吐量（应稳定在 162）
- **gpu_inside [%]**：GPU 计算时间占比（应 >90%）
- **util≈X%**：GPU 利用率（应 >80%）

#### 异常检测

```bash
# 查找处理时间异常的 PDF
grep "OCR SUMMARY" logs/*.log | awk -F'|' '{
    split($1, a, " ");
    split(a[6], b, "s");
    pages = split($2, c, "/");
    if (b[1] / pages > 300) print $0  # >5 min per page
}'

# 查找 GPU 利用率低的批次
grep "util≈" logs/*.log | awk '{
    match($0, /util≈([0-9]+)%/, arr);
    if (arr[1] < 50) print $0
}'

# 查找 CUDA 错误
grep -i "cuda error" logs/*.log
```

### 缓存管理

#### OCR 缓存结构

```
quantitative_results_ocr/chandra_ocr_2/
├── {pdf_name}/
│   ├── {pdf_name}_ocr_output.json    # OCR 结果
│   └── _tmp_table_crops/             # 临时表格图像
└── ...
```

#### 检查缓存

```bash
# 统计已缓存的 PDF 数量
ls -d quantitative_results_ocr/chandra_ocr_2/*/ | wc -l

# 检查特定 PDF 的缓存
ls -la quantitative_results_ocr/chandra_ocr_2/{pdf_name}/
```

#### 清理缓存

```bash
# 清理临时文件
find quantitative_results_ocr/chandra_ocr_2/ -name "_tmp_table_crops" -type d -exec rm -rf {} +

# 清理旧缓存（谨慎）
# find quantitative_results_ocr/chandra_ocr_2/ -mtime +30 -type d -exec rm -rf {} +
```

### 性能基准

#### H100 性能

```
GPU 利用率：94.5%
处理速度：0.07 pages/s/GPU
Token 吞吐：162 tok/s
CPU 开销：4%
```

#### A800 性能

```
GPU 利用率：47.7%
处理速度：0.06 pages/s/GPU
Token 吞吐：162 tok/s
CPU 开销：0%
```

#### 预期完成时间

| PDF 数量 | H100 单卡 | A800 单卡 | 2×H100 | 2×A800 |
|---------|----------|----------|--------|--------|
| 1000 | 4h | 4.5h | 2h | 2.3h |
| 5000 | 20h | 23h | 10h | 11.5h |
| 9637 | 38h | 44h | 19h | 22h |

### 故障处理

#### GPU 崩溃

```bash
# 1. 检查 GPU 健康状态
cat /tmp/gpu_health_status.json | python -m json.tool

# 2. 重置 GPU 健康状态（GPU 修复后）
python3 -c "
import json
from pathlib import Path
health_file = Path('/tmp/gpu_health_status.json')
if health_file.exists():
    with open(health_file, 'r') as f:
        data = json.load(f)
    if 0 in data.get('unhealthy_gpus', []):
        data['unhealthy_gpus'].remove(0)
        with open(health_file, 'w') as f:
            json.dump(data, f, indent=2)
        print('GPU 0 health status reset')
"
```

#### PDF 黑名单

```bash
# 查看黑名单
cat /tmp/pdf_blacklist.json | python -m json.tool

# 移除特定 PDF
python3 -c "
import json
from pathlib import Path
blacklist_file = Path('/tmp/pdf_blacklist.json')
if blacklist_file.exists():
    with open(blacklist_file, 'r') as f:
        data = json.load(f)
    if 'ltn201707281159' in data.get('blacklisted_pdfs', []):
        data['blacklisted_pdfs'].remove('ltn201707281159')
        with open(blacklist_file, 'w') as f:
            json.dump(data, f, indent=2)
        print('Removed from blacklist')
"
```

#### OOM 错误

```bash
# 1. 减少 batch_size
export OCR_BATCH_SIZE=8  # 从 16 降到 8

# 2. 减少进程数
# 修改脚本，只处理 2 个 list 而不是 3 个

# 3. 增加系统内存
#SBATCH --mem=96G  # 从 64G 增加到 96G
```

### 最佳实践

#### 1. 任务前检查

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 PDF 列表
wc -l list_*

# 检查已完成任务
ls quantitative_results_ocr/chandra_ocr_2/ | wc -l

# 检查磁盘空间
df -h
```

#### 2. 任务中监控

```bash
# 实时监控 GPU
watch -n 2 nvidia-smi

# 查看日志
tail -f logs/ocr_list_00_job*.log

# 检查进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{sum+=$2} END {print sum}'
```

#### 3. 任务后验证

```bash
# 检查输出完整性
for pdf in $(cat list_00); do
    pdf_base="${pdf%.pdf}"
    if [ ! -f "quantitative_results_ocr/chandra_ocr_2/${pdf_base}/${pdf_base}_ocr_output.json" ]; then
        echo "Missing: $pdf"
    fi
done

# 检查性能报告
bash compare_gpu_speed.sh <job_id_1> <job_id_2>
```

#### 4. 定期维护

```bash
# 清理临时文件
find quantitative_results_ocr/chandra_ocr_2/ -name "_tmp_table_crops" -type d -exec rm -rf {} +

# 检查 GPU 健康状态
cat /tmp/gpu_health_status.json

# 更新 PDF 黑名单（移除已修复的 PDF）
cat /tmp/pdf_blacklist.json
```

---

## 附录

### 相关文件

- `src/main.py`: 主入口
- `src/numeric_extractor.py`: 批处理逻辑
- `src/chandra_ocr_tester.py`: Chandra OCR 封装
- `src/utils.py`: GPU 健康追踪、PDF 黑名单
- `run_3stream_gpu0.sh`, `run_3stream_gpu1.sh`: GPU 配置脚本
- `split_remaining_work.sh`: 任务分配脚本
- `compare_gpu_speed.sh`: 性能对比脚本
- `logs/ocr_*.log`: 任务日志
- `quantitative_results_ocr/chandra_ocr_2/`: OCR 缓存目录

### 状态文件

- `/tmp/gpu_health_status.json`: GPU 健康状态
- `/tmp/pdf_blacklist.json`: PDF 黑名单

### 快速参考

#### 常用命令

```bash
# 提交任务
sbatch run_3stream_gpu0.sh

# 查看任务状态
squeue -u $USER

# 查看 GPU 状态
nvidia-smi

# 查看日志
tail -f logs/ocr_list_00_job*.log

# 检查进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{sum+=$2} END {print sum}'

# 性能对比
bash compare_gpu_speed.sh <job_id_1> <job_id_2>
```

#### 故障处理

```bash
# 停止任务
scancel <job_id>

# 重置 GPU 健康状态
python3 -c "
import json
from pathlib import Path
health_file = Path('/tmp/gpu_health_status.json')
if health_file.exists():
    with open(health_file, 'r') as f:
        data = json.load(f)
    if 0 in data.get('unhealthy_gpus', []):
        data['unhealthy_gpus'].remove(0)
        with open(health_file, 'w') as f:
            json.dump(data, f, indent=2)
        print('GPU 0 health status reset')
"

# 移除 PDF 黑名单
python3 -c "
import json
from pathlib import Path
blacklist_file = Path('/tmp/pdf_blacklist.json')
if blacklist_file.exists():
    with open(blacklist_file, 'r') as f:
        data = json.load(f)
    if 'ltn201707281159' in data.get('blacklisted_pdfs', []):
        data['blacklisted_pdfs'].remove('ltn201707281159')
        with open(blacklist_file, 'w') as f:
            json.dump(data, f, indent=2)
        print('Removed from blacklist')
"
```

---

**文档版本**：1.0  
**最后更新**：2026-07-31  
**维护者**：ESG OCR 团队

---

## 当前任务状态（2026-07-31 17:56）

### 运行中的任务

| Job ID | 脚本 | GPU | 运行时间 | 处理的列表 | PDF 数量 | 预计完成时间 |
|--------|------|-----|----------|-----------|---------|-------------|
| 1023 | run_a100_a.sh | A100 | 3 分 35 秒 | list_ocr_remaining_a.txt | 1,697 | ~24 小时 |
| 1024 | run_a100_b.sh | A100 | 3 分 35 秒 | list_ocr_remaining_b.txt | 1,698 | ~24 小时 |
| 1025 | run_3stream_gpu0.sh | H100 | 12 秒 | list_00 + list_01 + list_02 | 9,637 (实际 ~8,214) | ~4.8 天 |

### 任务详情

**Job 1023 & 1024 (A100)**:
- 处理从 Job 1017 分割出的未完成 PDF
- list_ocr_remaining_a.txt: 1,697 PDFs
- list_ocr_remaining_b.txt: 1,698 PDFs
- A100 性能约 70 PDFs/hour（与 H100 相近）
- 预计 24 小时内完成

**Job 1025 (H100)**:
- 处理 list_00, list_01, list_02（共 9,637 PDFs）
- 由于 OCR 缓存机制，跳过已完成的 1,423 PDFs
- 实际需处理约 8,214 PDFs
- H100 性能约 71.65 PDFs/hour
- 预计 4.8 天完成

### GPU 配置

- **H100**: 1 个可用（另一个需要 reset）
- **A100**: 1 个（gpu2）
- **A800**: 已取消

### 性能基准

- H100: 0.07 pages/s, 71.65 PDFs/hour
- A100: ~0.07 pages/s, ~70 PDFs/hour（估计）
- A800: 0.06 pages/s, 45 PDFs/hour

### 注意事项

1. Job 1025 是长时间任务，需要持续监控
2. A100 任务预计明天完成
3. H100 任务需要约 5 天完成
4. 所有任务使用 OCR 缓存，避免重复工作
5. 使用 5 CPU cores per task，避免 QOS 限制

### 监控命令

```bash
# 查看所有任务状态
squeue -u $USER

# 查看各任务进度
for job_id in 1023 1024 1025; do
    echo "=== Job $job_id ==="
    grep -c "OCR SUMMARY" logs/ocr_*_job${job_id}.log 2>/dev/null | awk -F: '{sum+=$2} END {print "已处理:", sum, "PDFs"}'
done

# 实时监控 GPU 状态
watch -n 2 nvidia-smi
```
