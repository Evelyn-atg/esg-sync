---
name: gpu-crash-diagnosis-special-pdfs
description: "GPU 崩溃诊断过程，特殊问题 PDF 分析（ltn201707281159, 2023042100201），崩溃原因和解决方案"
metadata: 
  node_type: memory
  type: project
  related: "[[gpu-error-handling-pdf-blacklist]], [[h100-a800-performance-comparison]]"
  originSessionId: 22294c7b-2e97-4645-a148-1a37910e37ce
  modified: 2026-07-31T09:10:33.664Z
---

# GPU 崩溃诊断和特殊 PDF 分析

## 崩溃事件

### 事件概述

**时间**：2026-07-30 07:03:05  
**Job**：1013 (GPU 0, H100)  
**运行时长**：5 小时 23 分钟（从 01:40:24 到 07:03:05）  
**崩溃后**：Job 继续运行 5+ 小时，但所有 OCR 失败，回退到 text_fallback

### 错误日志

```
2026-07-30 07:03:05,839 - WARNING - [esg-ocr][gpu=0][ltn201707281159] 
Batch 1/1 failed entirely (pages [1, 2]): CUDA error: unspecified launch failure
CUDA kernel errors might be asynchronously reported at some other API call, 
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

### 崩溃时间线

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

### 影响统计

- **成功 OCR**：362 个 PDF
- **OCR 失败**：7,436 个 PDF（回退到 text_fallback）
- **浪费计算时间**：约 5 小时 20 分

### GPU 0 当前状态

```bash
$ srun --jobid=1016 nvidia-smi
index, temperature.gpu, utilization.gpu [%], memory.used [MiB]
0, [GPU requires reset], [GPU requires reset], 0 MiB
```

**结论**：GPU 0 需要硬件重置，可能是硬件故障。

## 对比：GPU 1 正常运行

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

## 崩溃原因分析

### 可能的原因

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

### 诊断命令

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

### 向 IT 报告

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

## 特殊问题 PDF 分析

### 1. ltn201707281159.pdf

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

### 2. 2023042100201.pdf

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
3. **内存带宽瓶颈**：A800 的 NVLink 带宽限制
4. **模型过载**：单个 PDF 占用过多 GPU 资源

**处理**：
- 已加入 PDF 黑名单
- 建议：对于高分辨率 PDF，降低图像分辨率或跳过

### 3. 其他异常 PDF

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

## 解决方案

### 短期措施

1. **GPU 健康追踪**：已实现 GPUHealthTracker
2. **PDF 黑名单**：已实现 PDFBlacklist
3. **立即停止**：GPU 崩溃后立即停止 worker
4. **自动跳过**：问题 PDF 自动加入黑名单

### 中期优化

1. **PDF 预处理**：
   - 检测高分辨率图像
   - 自动降低分辨率
   - 跳过复杂 PDF

2. **动态批次调整**：
   - 检测 GPU 利用率
   - 自动调整 batch_size
   - 避免过载

3. **异常检测**：
   - 监控处理时间
   - 检测异常 PDF
   - 自动加入黑名单

### 长期改进

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

## 相关文件

- `src/utils.py`: GPUHealthTracker, PDFBlacklist
- `src/chandra_ocr_tester.py`: CUDA 错误捕获
- `src/numeric_extractor.py`: GPUFatalError 处理
- `/tmp/gpu_health_status.json`: GPU 健康状态
- `/tmp/pdf_blacklist.json`: PDF 黑名单
- `HKEX ESG Reports/ltn201707281159.pdf`: 问题 PDF 1
- `HKEX ESG Reports/2023042100201.pdf`: 问题 PDF 2
