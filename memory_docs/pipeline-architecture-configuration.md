---
name: pipeline-architecture-configuration
description: ESG OCR pipeline 架构，配置文件，关键参数，运行流程和最佳实践
metadata:
  type: reference
  related: [[gpu-batching-optimization]], [[memory-resource-management]], [[workload-distribution-strategies]]
---

# Pipeline 架构和配置

## 系统架构

### 整体流程

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

### 关键组件

```
src/
├── main.py                      # 主入口
├── numeric_extractor.py         # 数字提取器（批处理逻辑）
├── chandra_ocr_tester.py        # Chandra OCR 封装
├── utils.py                     # GPU 健康追踪、PDF 黑名单
└── config.py                    # 配置管理
```

### 数据流

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

## 配置文件

### SLURM 作业配置

```bash
#SBATCH -p H100                  # 分区：H100 或 A800
#SBATCH --gres=gpu:1             # GPU 数量
#SBATCH --cpus-per-task=8        # CPU cores per process
#SBATCH --mem=64G                # 系统内存（3 进程 × 20GB + 4GB）
#SBATCH -o ocr_%j.out            # 输出日志
```

### 环境变量

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

### Python 配置

```python
# src/numeric_extractor.py
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "16"))

# src/utils.py
_GPU_HEALTH_FILE = Path("/tmp/gpu_health_status.json")
_PDF_BLACKLIST_FILE = Path("/tmp/pdf_blacklist.json")
```

## 运行流程

### 1. 准备阶段

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 PDF 列表
wc -l list_*

# 检查已完成的任务
ls -la quantitative_results_ocr/chandra_ocr_2/ | wc -l
```

### 2. 分配任务

```bash
# 基于 OCR 缓存分配
bash split_remaining_work.sh

# 查看分配结果
wc -l list_ocr_remaining_*.txt
```

### 3. 提交任务

```bash
# 单 GPU
sbatch run_3stream_gpu0.sh

# 多 GPU
sbatch run_3stream_gpu0.sh
sbatch run_3stream_gpu1.sh
```

### 4. 监控任务

```bash
# 查看任务状态
squeue -u $USER

# 查看 GPU 状态
watch -n 2 nvidia-smi

# 查看日志
tail -f logs/ocr_list_00_job*.log
```

### 5. 检查进度

```bash
# 统计已处理的 PDF
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{sum+=$2} END {print sum}'

# 检查 GPU 性能
bash compare_gpu_speed.sh <job_id_1> <job_id_2>
```

## 关键参数

### Batch Size

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 单进程/GPU | 16 | 最大化 GPU 利用率 |
| 2 进程/GPU | 12 | 平衡利用率和显存 |
| 3 进程/GPU | 8 | 避免 OOM |
| 4+ 进程/GPU | 4-6 | 显存限制 |

### 进程数

| GPU 类型 | 推荐进程数 | 说明 |
|----------|-----------|------|
| H100 | 3 | GPU 利用率 94.5% |
| A800 | 3 | GPU 利用率 47.7%（受内存带宽限制） |

### 系统内存

| 进程数 | 推荐内存 | 说明 |
|--------|---------|------|
| 1 | 32G | 单进程 |
| 2 | 48G | 双进程 |
| 3 | 64G | 推荐配置 |
| 4+ | 96G+ | 高并发 |

## 日志分析

### OCR 日志格式

```
[时间戳] - [模块] - INFO - [esg-ocr][gpu=<id>][<pdf_name>] 
OCR batching: <pages> pages → <batches> batch(es) of ≤<batch_size>

[时间戳] - [模块] - INFO - [esg-ocr][gpu=<id>][<pdf_name>] 
Batch <idx>/<total> done (pages <start>-<end>, <tokens> tok): 
  wall <time>s = img_load <time>s (<pct>%) + generate <time>s 
    (gpu <time>s [<pct>%], cpu_overhead <time>s [<pct>%]) + postproc <time>s (<pct>%) |
  <pages_per_s> pages/s, <tokens_per_s> tok/s, util≈<util>%
```

### 关键指标

- **pages/s**：实际处理速度（应 >0.05）
- **tokens/s**：模型计算吞吐量（应稳定在 162）
- **gpu_inside [%]**：GPU 计算时间占比（应 >90%）
- **util≈X%**：GPU 利用率（应 >80%）

### 异常检测

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

## 缓存管理

### OCR 缓存结构

```
quantitative_results_ocr/chandra_ocr_2/
├── {pdf_name}/
│   ├── {pdf_name}_ocr_output.json    # OCR 结果
│   └── _tmp_table_crops/             # 临时表格图像
└── ...
```

### 检查缓存

```bash
# 统计已缓存的 PDF 数量
ls -d quantitative_results_ocr/chandra_ocr_2/*/ | wc -l

# 检查特定 PDF 的缓存
ls -la quantitative_results_ocr/chandra_ocr_2/{pdf_name}/
```

### 清理缓存

```bash
# 清理临时文件
find quantitative_results_ocr/chandra_ocr_2/ -name "_tmp_table_crops" -type d -exec rm -rf {} +

# 清理旧缓存（谨慎）
# find quantitative_results_ocr/chandra_ocr_2/ -mtime +30 -type d -exec rm -rf {} +
```

## 性能基准

### H100 性能

```
GPU 利用率：94.5%
处理速度：0.07 pages/s/GPU
Token 吞吐：162 tok/s
CPU 开销：4%
```

### A800 性能

```
GPU 利用率：47.7%
处理速度：0.06 pages/s/GPU
Token 吞吐：162 tok/s
CPU 开销：0%
```

### 预期完成时间

| PDF 数量 | H100 单卡 | A800 单卡 | 2×H100 | 2×A800 |
|---------|----------|----------|--------|--------|
| 1000 | 4h | 4.5h | 2h | 2.3h |
| 5000 | 20h | 23h | 10h | 11.5h |
| 9637 | 38h | 44h | 19h | 22h |

## 故障处理

### GPU 崩溃

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

### PDF 黑名单

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

### OOM 错误

```bash
# 1. 减少 batch_size
export OCR_BATCH_SIZE=8  # 从 16 降到 8

# 2. 减少进程数
# 修改脚本，只处理 2 个 list 而不是 3 个

# 3. 增加系统内存
#SBATCH --mem=96G  # 从 64G 增加到 96G
```

## 最佳实践

### 1. 任务前检查

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

### 2. 任务中监控

```bash
# 实时监控 GPU
watch -n 2 nvidia-smi

# 查看日志
tail -f logs/ocr_list_00_job*.log

# 检查进度
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{sum+=$2} END {print sum}'
```

### 3. 任务后验证

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

### 4. 定期维护

```bash
# 清理临时文件
find quantitative_results_ocr/chandra_ocr_2/ -name "_tmp_table_crops" -type d -exec rm -rf {} +

# 检查 GPU 健康状态
cat /tmp/gpu_health_status.json

# 更新 PDF 黑名单（移除已修复的 PDF）
cat /tmp/pdf_blacklist.json
```

## 相关文件

- `src/main.py`: 主入口
- `src/numeric_extractor.py`: 批处理逻辑
- `src/chandra_ocr_tester.py`: Chandra OCR 封装
- `src/utils.py`: GPU 健康追踪、PDF 黑名单
- `run_3stream_gpu0.sh`, `run_3stream_gpu1.sh`: GPU 配置脚本
- `split_remaining_work.sh`: 任务分配脚本
- `compare_gpu_speed.sh`: 性能对比脚本
- `logs/ocr_*.log`: 任务日志
- `quantitative_results_ocr/chandra_ocr_2/`: OCR 缓存目录
