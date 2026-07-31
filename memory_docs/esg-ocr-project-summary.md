---
name: esg-ocr-project-summary
description: ESG OCR 项目总结，包括所有优化、问题、解决方案和最佳实践的综合索引
metadata: 
  node_type: memory
  type: project
  related: "[[gpu-batching-optimization]], [[gpu-error-handling-pdf-blacklist]], [[h100-a800-performance-comparison]], [[gpu-crash-diagnosis-special-pdfs]], [[workload-distribution-strategies]], [[memory-resource-management]], [[pipeline-architecture-configuration]]"
  originSessionId: 22294c7b-2e97-4645-a148-1a37910e37ce
  modified: 2026-07-31T09:12:35.545Z
---

# ESG OCR 项目总结

## 项目概述

**目标**：使用 Chandra OCR 处理 9637 份 ESG 报告的表格页，提取结构化数字数据。

**时间线**：2026-07-29 至 2026-07-31

**关键技术**：
- Chandra OCR（9.9GB 模型）
- 多 GPU 并行处理（H100 + A800）
- 批处理优化（batch=16）
- GPU 错误处理和 PDF 黑名单
- 动态工作负载分配

## 核心改进

### 1. GPU 批处理优化

**问题**：单张处理 GPU 利用率仅 40-60%

**解决方案**：实现 batch=16 批处理

**效果**：
- GPU 利用率：40-60% → 94.5%（H100）
- 处理速度：提升 1.5-2 倍
- tokens/s：稳定在 162

**相关文件**：[[gpu-batching-optimization]]

### 2. GPU 错误处理和 PDF 黑名单

**问题**：GPU 崩溃后继续运行 5+ 小时，所有 OCR 失败

**解决方案**：
- GPUHealthTracker：跨进程 GPU 健康状态共享
- PDFBlacklist：自动跳过问题 PDF
- GPUFatalError：立即停止 worker

**效果**：
- 崩溃后立即停止（几秒钟 vs 5+ 小时）
- 自动标记不健康 GPU
- 问题 PDF 自动加入黑名单

**相关文件**：[[gpu-error-handling-pdf-blacklist]]

### 3. 工作负载分配

**问题**：工作负载不均衡，某些 GPU 提前完成

**解决方案**：基于 OCR 缓存的动态分配

**效果**：
- 无重复工作
- 支持断点续传
- 预计完成时间从 88 小时减半到 37.7 小时

**相关文件**：[[workload-distribution-strategies]]

### 4. 内存和资源管理

**问题**：OOM 错误、性能下降

**解决方案**：
- 合理的 batch_size 配置
- 适当的进程数（3 进程/GPU）
- 充足的系统内存（64GB）

**效果**：
- 避免 OOM
- 最大化 GPU 利用率
- 稳定运行

**相关文件**：[[memory-resource-management]]

## 性能数据

### H100 vs A800

| 指标 | H100 | A800 |
|------|------|------|
| GPU 利用率 | 94.5% | 47.7% |
| 处理速度 | 0.07 pages/s | 0.06 pages/s |
| Token 吞吐 | 162 tok/s | 162 tok/s |
| CPU 开销 | 4% | 0% |

**结论**：H100 性能更优，优先使用 H100 处理大量任务

**相关文件**：[[h100-a800-performance-comparison]]

## 特殊问题 PDF

### 1. ltn201707281159

- **股票**：01622 REDCO PPT
- **问题**：触发 GPU 0 崩溃
- **特征**：5 页，160KB，无明显异常
- **处理**：已加入黑名单

### 2. 2023042100201

- **股票**：02382 SUNNY OPTICAL
- **问题**：在 A800 上处理极慢（1 页花 12 分钟）
- **特征**：91 页，12MB，高分辨率图像（2598×3484）
- **处理**：已加入黑名单

**相关文件**：[[gpu-crash-diagnosis-special-pdfs]]

## GPU 崩溃事件

**时间**：2026-07-30 07:03:05  
**Job**：1013 (GPU 0, H100)  
**原因**：可能是硬件故障

**影响**：
- 浪费 5+ 小时计算时间
- 7,436 个 PDF 需要重新处理

**解决方案**：
- 实现 GPU 健康追踪
- 实现 PDF 黑名单
- 向 IT 报告并请求 GPU 重置

**相关文件**：[[gpu-crash-diagnosis-special-pdfs]], [[gpu-error-handling-pdf-blacklist]]

## 配置模板

### 单 GPU 配置

```bash
#!/bin/bash
#SBATCH -p H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export OCR_BATCH_SIZE=16
python -m src.main --step numeric_extraction --pdf_list_file list_00
```

### 多 GPU 配置（3 进程/GPU）

```bash
#!/bin/bash
#SBATCH -p H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

export OCR_BATCH_SIZE=8
for lst in list_00 list_01 list_02; do
    python -m src.main --step numeric_extraction --pdf_list_file "$lst" &
done
wait
```

**相关文件**：[[pipeline-architecture-configuration]], [[memory-resource-management]]

## 监控命令

### 实时监控

```bash
# GPU 状态
watch -n 2 nvidia-smi

# 任务进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{sum+=$2} END {print sum}'

# 性能对比
bash compare_gpu_speed.sh <job_id_1> <job_id_2>
```

### 故障排查

```bash
# 检查 GPU 健康状态
cat /tmp/gpu_health_status.json | python -m json.tool

# 检查 PDF 黑名单
cat /tmp/pdf_blacklist.json | python -m json.tool

# 检查异常 PDF
grep "OCR SUMMARY" logs/*.log | awk -F'|' '{
    split($1, a, " ");
    split(a[6], b, "s");
    pages = split($2, c, "/");
    if (b[1] / pages > 300) print $0
}'
```

**相关文件**：[[pipeline-architecture-configuration]]

## 最佳实践

### 任务前

1. 检查 GPU 状态：`nvidia-smi`
2. 检查 PDF 列表：`wc -l list_*`
3. 检查已完成任务：`ls quantitative_results_ocr/chandra_ocr_2/ | wc -l`
4. 基于缓存分配任务：`bash split_remaining_work.sh`

### 任务中

1. 实时监控 GPU：`watch -n 2 nvidia-smi`
2. 查看日志：`tail -f logs/ocr_list_00_job*.log`
3. 检查进度：定期运行进度统计命令

### 任务后

1. 检查输出完整性
2. 生成性能报告
3. 清理临时文件
4. 更新 PDF 黑名单（移除已修复的 PDF）

**相关文件**：[[pipeline-architecture-configuration]]

## 经验教训

### 1. GPU 批处理至关重要

- 单张处理浪费 40-60% GPU 时间
- 批处理可以提升到 95% 利用率
- 实际速度提升 1.5-2 倍

### 2. 错误处理不可或缺

- GPU 崩溃后必须立即停止
- 问题 PDF 必须自动跳过
- 跨进程状态共享很重要

### 3. 工作负载均衡

- 均匀分配任务避免浪费
- 基于缓存的动态分配最灵活
- 支持断点续传很实用

### 4. 内存管理要谨慎

- 合理的 batch_size 和进程数
- 充足的系统内存（64GB）
- 避免 OOM 和性能下降

### 5. H100 vs A800

- H100 利用率更高（94.5% vs 47.7%）
- Token 吞吐量相同（162 tok/s）
- 优先使用 H100 处理大量任务

## 未来改进

### 短期

1. **PDF 预处理**：自动检测高分辨率图像并降低分辨率
2. **动态批次调整**：根据 GPU 利用率自动调整 batch_size
3. **异常检测**：自动识别和跳过问题 PDF

### 中期

1. **PDF 质量评估**：预处理阶段评估 PDF 复杂度
2. **GPU 负载均衡**：自动切换到健康 GPU
3. **模型优化**：与 Chandra 团队反馈问题

### 长期

1. **分布式处理**：跨多个节点的处理
2. **智能调度**：基于 PDF 复杂度的智能分配
3. **性能预测**：预测任务完成时间

## 相关文件索引

### 核心文件

- `src/main.py`: 主入口
- `src/numeric_extractor.py`: 批处理逻辑
- `src/chandra_ocr_tester.py`: Chandra OCR 封装
- `src/utils.py`: GPU 健康追踪、PDF 黑名单

### 配置文件

- `run_3stream_gpu0.sh`, `run_3stream_gpu1.sh`: GPU 配置脚本
- `split_remaining_work.sh`: 任务分配脚本
- `compare_gpu_speed.sh`: 性能对比脚本

### 数据文件

- `HKEX ESG Reports/`: PDF 文件目录
- `quantitative_results_ocr/chandra_ocr_2/`: OCR 缓存目录
- `logs/ocr_*.log`: 任务日志

### 状态文件

- `/tmp/gpu_health_status.json`: GPU 健康状态
- `/tmp/pdf_blacklist.json`: PDF 黑名单

## 快速参考

### 常用命令

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

### 故障处理

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

## 总结

ESG OCR 项目通过以下关键改进实现了高效、稳定的批处理：

1. **GPU 批处理优化**：提升 1.5-2 倍速度
2. **GPU 错误处理**：避免浪费计算时间
3. **工作负载均衡**：最大化资源利用率
4. **内存管理**：避免 OOM 和性能下降

这些改进使项目能够稳定处理 9637 份 ESG 报告，预计总耗时从原来的数天缩短到约 38 小时（2×H100）。

---

**文档版本**：1.0  
**最后更新**：2026-07-31  
**维护者**：ESG OCR 团队
