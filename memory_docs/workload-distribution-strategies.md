---
name: workload-distribution-strategies
description: 多 GPU 工作负载分配策略，均匀分配任务，避免重复工作，最大化效率
metadata: 
  node_type: memory
  type: project
  related: "[[gpu-batching-optimization]], [[h100-a800-performance-comparison]]"
  originSessionId: 22294c7b-2e97-4645-a148-1a37910e37ce
  modified: 2026-07-31T09:08:22.880Z
---

# 工作负载分配策略

## 问题背景

在运行 9637 份 ESG 报告 OCR 任务时，需要合理分配工作负载到多个 GPU，避免：
- 重复工作（浪费计算资源）
- 工作不均衡（某些 GPU 提前完成，其他 GPU 仍在运行）
- 缓存冲突（多个进程同时写入同一文件）

## 策略概述

### 1. 基于 OCR 缓存的分配（推荐）

**原理**：利用 OCR 缓存文件判断哪些 PDF 已完成，只分配未完成的 PDF。

**优点**：
- ✅ 无重复工作
- ✅ 支持动态添加 GPU
- ✅ 支持断点续传

**缺点**：
- ⚠️ 需要扫描缓存目录
- ⚠️ 缓存文件可能被清理

### 2. 预分割列表

**原理**：在任务开始前，将 PDF 列表均匀分割成多个子列表。

**优点**：
- ✅ 简单直接
- ✅ 无需扫描缓存

**缺点**：
- ❌ 工作可能不均衡（某些 PDF 处理时间长）
- ❌ 不支持动态调整

### 3. 动态负载均衡

**原理**：根据 GPU 实时利用率动态分配任务。

**优点**：
- ✅ 最大化 GPU 利用率
- ✅ 自动适应不同 PDF 复杂度

**缺点**：
- ⚠️ 实现复杂
- ⚠️ 需要额外的调度逻辑

## 实现方案

### 方案 1：基于 OCR 缓存的分配

#### 步骤 1：扫描已完成的 PDF

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

#### 步骤 2：创建分配脚本

```bash
# GPU 0 处理 list_ocr_remaining_a.txt
cat > run_a800_a.sh << 'EOF'
#!/bin/bash
#SBATCH -p A800
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

echo "launching OCR stream for list_ocr_remaining_a (job $JOB_TAG, A800)"
python -m src.main --step numeric_extraction --pdf_list_file list_ocr_remaining_a.txt --force > "logs/ocr_list_ocr_remaining_a_job${JOB_TAG}.log" 2>&1

echo "=== A800 done (job $JOB_TAG) ==="
EOF

chmod +x run_a800_a.sh

# GPU 1 处理 list_ocr_remaining_b.txt
cat > run_a800_b.sh << 'EOF'
#!/bin/bash
#SBATCH -p A800
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

echo "launching OCR stream for list_ocr_remaining_b (job $JOB_TAG, A800 new)"
python -m src.main --step numeric_extraction --pdf_list_file list_ocr_remaining_b.txt --force > "logs/ocr_list_ocr_remaining_b_job${JOB_TAG}.log" 2>&1

echo "=== A800 new done (job $JOB_TAG) ==="
EOF

chmod +x run_a800_b.sh
```

#### 步骤 3：提交任务

```bash
sbatch run_a800_a.sh
sbatch run_a800_b.sh
```

### 方案 2：预分割列表

#### 统计每个 list 的工作量

```bash
# 统计每个 list 的表格页数
for lst in list_*; do
    count=$(grep "Detected.*table pages" logs/ocr_${lst}_job*.log | wc -l)
    echo "$lst: $count pages"
done
```

#### 均匀分配

```bash
# 假设 list_00-05 共 9637 个 PDF
# 分成 6 份，每份约 1606 个 PDF

# 合并所有 PDF
cat list_00 list_01 list_02 list_03 list_04 list_05 > all_pdfs.txt

# 分割成 6 份
split -l 1606 -d -a 2 all_pdfs.txt list_new_

# 重命名
mv list_new_00 list_00
mv list_new_01 list_01
mv list_new_02 list_02
mv list_new_03 list_03
mv list_new_04 list_04
mv list_new_05 list_05

rm all_pdfs.txt
```

### 方案 3：动态负载均衡（高级）

```python
# 伪代码示例
class DynamicLoadBalancer:
    def __init__(self, gpu_list):
        self.gpu_queue = {gpu: [] for gpu in gpu_list}
    
    def assign_task(self, pdf_name, estimated_time):
        # 找到当前负载最轻的 GPU
        min_load_gpu = min(self.gpu_queue.keys(), 
                          key=lambda gpu: sum(self.gpu_queue[gpu]))
        self.gpu_queue[min_load_gpu].append(estimated_time)
        return min_load_gpu
    
    def get_tasks_for_gpu(self, gpu):
        return self.gpu_queue[gpu]
```

## 监控进度

### 检查每个 GPU 的进度

```bash
# 查看所有 A800 任务的进度
echo "=== 所有 A800 任务 ==="
for job_id in $(squeue -u $USER | grep A800 | awk '{print $1}'); do
    echo "Job $job_id:"
    grep -c "OCR SUMMARY" logs/ocr_list_*_job${job_id}.log 2>/dev/null | awk -F: '{sum+=$2} END {print "  已处理:", sum, "个 PDF"}'
done

# 总体进度
total=4818
done=$(grep -c "OCR SUMMARY" logs/ocr_list_*_job1017.log logs/ocr_list_*_job<NEW_JOB_ID>.log 2>/dev/null | awk -F: '{sum+=$2} END {print sum}')
echo "=== list_00, 01, 02 总进度 ==="
echo "已完成: $done / $total ($((done * 100 / total))%)"
```

### 检查 H100 和 A800 的对比

```bash
echo "=== H100 进度 ==="
grep -c "OCR SUMMARY" logs/ocr_list_*_job1014.log | awk -F: '{sum+=$2} END {print sum, "个 PDF"}'

echo "=== A800 进度 ==="
grep -c "OCR SUMMARY" logs/ocr_list_*_job1017.log | awk -F: '{sum+=$2} END {print sum, "个 PDF"}'
```

## 实际案例

### 2026-07-30 任务分配

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
   - 平均分配给两张 A800

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
   - 分配给两张 A800：1697 + 1698
   - 预计完成时间：37.7 小时（从 88 小时减半）

## 最佳实践

### 1. 任务前评估

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

### 2. 动态调整

```bash
# 监控 GPU 利用率
watch -n 5 "srun --jobid=1017 nvidia-smi --query-gpu=utilization.gpu --format=csv"

# 如果利用率持续低于 50%，考虑调整 batch_size
# 如果利用率持续高于 95%，考虑减少 batch_size
```

### 3. 断点续传

```bash
# 检查哪些 PDF 未完成
bash split_remaining_work.sh

# 重新提交未完成的任务
sbatch run_a800_a.sh
sbatch run_a800_b.sh
```

## 相关文件

- `split_remaining_work.sh`: 扫描并分割未完成的 PDF
- `run_a800_a.sh`, `run_a800_b.sh`: A800 任务脚本
- `logs/ocr_list_*_job*.log`: 任务日志
- `quantitative_results_ocr/chandra_ocr_2/`: OCR 缓存目录
