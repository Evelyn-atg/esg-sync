---
name: memory-resource-management
description: GPU 显存和系统内存管理，避免 OOM，优化资源配置，batch size 与进程数的平衡
metadata:
  type: reference
  related: [[gpu-batching-optimization]], [[chandra-gpu-cluster-setup]]
---

# 内存和资源管理

## 问题背景

Chandra OCR 批处理任务需要大量 GPU 显存和系统内存，不当的配置会导致：
- GPU OOM（Out of Memory）
- 系统 OOM Killer 杀死进程
- 性能下降（频繁 swap）
- 任务失败

## 显存使用分析

### 模型显存占用

```bash
# 检查模型大小
ls -lh ~/models/chandra-ocr-2/model.safetensors
# 输出：9.9G
```

**模型加载到 GPU 后占用**：~10GB

### 批处理显存占用

**每个 batch 的显存使用**：

```python
# batch_size=16 时的显存使用
# - 输入图像：16 × 3MB = 48MB
# - 中间激活：~2-5GB（取决于图像复杂度）
# - KV Cache：~1-3GB（生成长度 500-5000 tokens）
# 总计：~3-8GB per batch
```

### 进程显存占用

**每个 Python 进程的显存使用**：

```
模型权重：~10GB
PyTorch 开销：~2-3GB
Batch 处理：~3-8GB（取决于 batch_size）
总计：~15-21GB per process
```

### 多进程显存使用

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

## 系统内存管理

### 系统内存需求

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

### SLURM 内存配置

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

## Batch Size 配置

### 推荐配置

```bash
# 单进程模式（测试或小批量）
export OCR_BATCH_SIZE=16

# 多进程模式（3 streams per GPU）
export OCR_BATCH_SIZE=8
```

### 选择依据

| 场景 | 推荐 batch_size | 原因 |
|------|----------------|------|
| 单进程/GPU | 16 | 最大化 GPU 利用率 |
| 2 进程/GPU | 12 | 平衡利用率和显存 |
| 3 进程/GPU | 8 | 避免 OOM |
| 4+ 进程/GPU | 4-6 | 显存限制 |

### 动态调整

```python
# 检测 GPU 利用率和显存使用
import torch

def adjust_batch_size(gpu_id, current_batch_size):
    gpu_util = torch.cuda.utilization(gpu_id)
    mem_used = torch.cuda.memory_allocated(gpu_id) / 1e9  # GB
    mem_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
    
    # 如果利用率低且显存充足，增加 batch_size
    if gpu_util < 80 and mem_used < mem_total * 0.7:
        return min(current_batch_size * 2, 32)
    
    # 如果利用率高或显存紧张，减少 batch_size
    if gpu_util > 95 or mem_used > mem_total * 0.8:
        return max(current_batch_size // 2, 4)
    
    return current_batch_size
```

## 进程数配置

### 推荐配置

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

### 多 GPU 配置

```bash
# 2 GPU 节点
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16  # 每个 GPU 8 cores

# 6 进程（每个 GPU 3 进程）
python -m src.main --step numeric_extraction --pdf_list_file list_00 &
python -m src.main --step numeric_extraction --pdf_list_file list_01 &
python -m src.main --step numeric_extraction --pdf_list_file list_02 &
CUDA_VISIBLE_DEVICES=1 python -m src.main --step numeric_extraction --pdf_list_file list_03 &
CUDA_VISIBLE_DEVICES=1 python -m src.main --step numeric_extraction --pdf_list_file list_04 &
CUDA_VISIBLE_DEVICES=1 python -m src.main --step numeric_extraction --pdf_list_file list_05 &
wait
```

## 监控命令

### 实时监控 GPU 状态

```bash
# 监控 GPU 利用率和显存
watch -n 2 nvidia-smi

# 监控特定 GPU
watch -n 2 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"
```

### 监控进程显存使用

```bash
# 查看所有 Python 进程的显存使用
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv

# 查看特定进程的详细信息
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv | grep <PID>
```

### 监控系统内存

```bash
# 实时监控系统内存
watch -n 2 free -h

# 查看进程的内存使用
ps aux | grep python | awk '{print $2, $4, $6}' | column -t
# 列：PID, %MEM, RSS (KB)
```

### 检查 OOM 事件

```bash
# 查看 dmesg 中的 OOM 事件
dmesg | grep -i "out of memory" | tail -10

# 查看系统日志
journalctl -k | grep -i "oom" | tail -10
```

## 优化技巧

### 1. 预加载模型

```python
# 在脚本开始时加载模型，避免重复加载
ocr_tester = ChandraOCRTester()
ocr_tester._ensure_chandra_manager()
```

**优点**：
- ✅ 减少模型加载时间（每次 ~1 分钟）
- ✅ 避免显存碎片

### 2. 清理缓存

```bash
# 清理 PyTorch 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 清理系统缓存（需要 root）
sync; echo 3 > /proc/sys/vm/drop_caches
```

### 3. 限制 CPU 使用

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

**原因**：
- 避免 CPU 过度竞争
- 减少内存带宽压力
- 提高 GPU 利用率

### 4. 使用 swap 文件（临时方案）

```bash
# 创建 swap 文件（需要 root）
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 检查 swap 使用
swapon --show
```

**警告**：
- ⚠️ Swap 会显著降低性能
- ⚠️ 仅作为临时解决方案
- ⚠️ 应该优化配置而不是依赖 swap

## 故障排查

### GPU OOM

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

### 系统 OOM

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

### 性能下降

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

## 配置模板

### 单 GPU 配置

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

### 多 GPU 配置（3 进程/GPU）

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

## 相关文件

- `run_3stream_gpu0.sh`, `run_3stream_gpu1.sh`: GPU 配置脚本
- `src/numeric_extractor.py`: OCR_BATCH_SIZE 配置
- `logs/ocr_*.log`: 任务日志（检查 OOM 错误）
- `nvidia-smi`: GPU 监控工具
- `free`, `top`, `ps`: 系统内存监控工具
