# ESG OCR 项目文档

> Chandra OCR 批处理处理 9637 份 ESG 报告表格页，提取结构化数字数据。
> 本文档**只覆盖 OCR 阶段**（表格检测 → Chandra OCR → 缓存），数字提取（`numeric_extracts/`、`numeric_blocks.json`）不在本文范围内。
>
> **范围**：2026-07-29 ~ 2026-07-31 ｜ **模型**：Chandra OCR 9.9GB ｜ **当前算力**：A100（gpu2）+ A800（gpu1）

---

## 1. 核心结论

- **批处理是关键优化**：单张处理 GPU 利用率仅 40–60%；改为 batch=16 后利用率显著提升、整体提速 1.5–2×。Token 吞吐恒为 ~162 tok/s（模型算力不变，优化只是减少空闲）。
- **H100 当前不可用**：gpu3 节点 GPU 0（Bus `00000000:23:00.0`）崩溃且需硬件重置，Slurm 无健康感知持续把任务分到坏卡。已全面改用 A100 / A800（详见 §4）。
- **崩溃快速止损**：GPU 一旦报 CUDA 错误立即停 worker + 标记坏卡 + 问题 PDF 入黑名单（§3），避免重蹈 Job 1013 空跑 5 小时的覆辙。
- **工作负载靠缓存去重**：以 `chandra_ocr_2/` 缓存是否存在判完成，按剩余量动态分配，天然支持断点续传（§5）。

---

## 2. GPU 批处理优化

`src/numeric_extractor.py` 中按页批次送 GPU，而非逐张：

```python
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "16"))
for batch_start in range(0, len(valid_pages), batch_size):
    batch_images = [pdf_image_dir / f"page_{p:03d}.png"
                    for p in valid_pages[batch_start:batch_start + batch_size]]
    batch_results = ocr_tester._run_page_ocr_batch(batch_images)
```

**效果（实测）**

| 指标 | 优化前（单张） | 优化后（batch=16） |
|------|---------------|-------------------|
| GPU 利用率 | 40–60% | 显著提升（批处理消除空闲等待） |
| 速度 | ~0.03 pages/s | 0.06–0.07 pages/s |
| token 吞吐 | 162 tok/s | 162 tok/s（不变） |

**批大小与进程数（同一 GPU 上 N 个 stream 时向下调）**

| 进程数 / GPU | 推荐 batch | 备注 |
|------------|-----------|------|
| 1 | 16 | 单进程利用率最高（当前默认） |
| 2 | 12 | 平衡利用率与显存 |
| 3 | 8 | 原设计：每 job 3 stream |
| 4+ | 4–6 | 显存受限 |

> 小 PDF（仅 1–3 页）实际 batch 偏小，利用率会下降，无法避免，整体影响小。

**遥测日志关键字段**：`pages/s`（实际速度）、`tok/s`（应恒为 162）、`gpu_inside [X%]`（GPU 计算时间占比，应 >90%）、`util≈X%`（nvidia-smi 采样值）。

---

## 3. 错误处理：GPU 健康追踪 + PDF 黑名单

实现于 `src/utils.py`，跨进程通过本地文件共享状态：

- **GPUHealthTracker** → `/tmp/gpu_health_status.json`：`mark_gpu_unhealthy(gpu_id, err)` / `is_gpu_healthy(gpu_id)`。崩溃时记录坏卡。
- **PDFBlacklist** → `/tmp/pdf_blacklist.json`：`add_to_blacklist(pdf, reason)` / `is_blacklisted(pdf)`。问题 PDF 自动跳过。

崩溃捕获（`src/chandra_ocr_tester.py`）：捕获 `RuntimeError` 且含 `CUDA` 时，标记坏卡 → 该 PDF 入黑名单 → `raise GPUFatalError`。`numeric_extractor.py` 收到 `GPUFatalError` 后 `sampler.stop()` + 立即退出。

**管理命令**

```bash
cat /tmp/gpu_health_status.json | python -m json.tool   # 看坏卡
cat /tmp/pdf_blacklist.json      | python -m json.tool   # 看黑名单

# GPU 修复后重置（示例：移出 GPU 0）
python3 - <<'EOF'
import json
from pathlib import Path
f = Path("/tmp/gpu_health_status.json")
d = json.loads(f.read_text())
d["unhealthy_gpus"] = [g for g in d.get("unhealthy_gpus", []) if g != 0]
f.write_text(json.dumps(d, indent=2))
EOF
```

---

## 4. H100 不可用说明（2026-07-31）

**现象**：gpu3 节点 GPU 0（Bus `00000000:23:00.0`）运行约 5 小时后报 `CUDA error: unspecified launch failure`，随后 `nvidia-smi` 显示 `GPU requires reset`，该卡无法继续使用。Job 1025 踩中后三个 stream 零产出。

**根因（非配置/代码问题）**

1. **坏卡是唯一能摸到的 H100**：`ssh gpu3` 实测仅枚举到 1 张 H100，即坏掉的 GPU 0；文档中「GPU 1 正常（Job 1014）」对应的另一张 H100 在其他节点，本任务用不到。
2. **Slurm 无健康感知**：`scontrol show node gpu3` 仍显示 2 卡可用，调度器持续把新 H100 任务分到坏卡。

**当前决策**：放弃 H100，全部改用 A 卡——A100 分区（gpu2）跑 `list_a`/`list_b`，A800 分区（gpu1）跑 `list_c`，待 IT 修复 GPU 0 后再评估。

> `nvidia-smi` 等 GPU 命令须在计算节点执行：`ssh gpu3 "nvidia-smi --query-gpu=index,name,pci.bus_id,uuid --format=csv"`（登录节点无此命令）。

---

## 5. 工作负载分配（基于 OCR 缓存）

利用缓存文件判完成，避免重复与不均衡：

```bash
# 未完成 OCR 的 PDF（无缓存即未完成）
for pdf in $(cat list_00 list_01 list_02); do
  b="${pdf%.pdf}"
  [ -f "quantitative_results_ocr/chandra_ocr_2/$b/${b}_ocr_output.json" ] || echo "$pdf"
done > list_ocr_remaining.txt
# 按需均分后 sbatch 提交
```

**优点**：无重复、支持动态加卡、支持断点续传。**缺点**：需扫缓存目录。

**进度统计**

```bash
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{s+=$2} END{print s, "PDFs done"}'
squeue -u $USER
```

---

## 6. 资源与显存管理

**显存**：模型权重 ~10GB + PyTorch 开销 2–3GB + batch 3–8GB ≈ **15–21GB / 进程**。80GB 卡上 3 进程 ~63GB 安全；4 进程有 OOM 风险。

**系统内存**：每进程 ~15–20GB（含 PDF 解析），故 `--mem=64G`（3 进程）。此前 32G 会触发 OOM Killer。

**当前作业配置（已验证可用）**

```bash
#SBATCH --gres=gpu:1          # 每 job 1 张卡
#SBATCH --cpus-per-task=5      # QoS 限 16 CPU/用户，3 job × 5 = 15 ≤ 16
#SBATCH --mem=64G
export OCR_BATCH_SIZE=16       # 单 stream
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2  # 限制 CPU 线程，避免竞争
```

**监控**

```bash
watch -n 2 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv"
watch -n 2 free -h                       # 系统内存
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
```

**故障排查**：GPU OOM → 降 `OCR_BATCH_SIZE` 或减进程；系统 OOM → 加 `--mem`；利用率 <50% → 查 batch / CPU 瓶颈（`top`）/ I/O（`iostat -x 1`）。

---

## 7. Pipeline 架构

```
PDF → 表格页检测（文本特征） → Chandra OCR 批处理（GPU） → HTML/Markdown → 缓存
```

**组件**（`src/`）

| 文件 | 职责 |
|------|------|
| `main.py` | 主入口（`--step numeric_extraction --pdf_list_file <list> --force`） |
| `numeric_extractor.py` | 数字/表格提取 + 批处理逻辑 |
| `chandra_ocr_tester.py` | Chandra OCR 封装 + CUDA 错误捕获 |
| `utils.py` | GPU 健康追踪、PDF 黑名单 |

**输出目录**

```
quantitative_results_ocr/chandra_ocr_2/
└── {pdf_name}/
    ├── {pdf_name}_ocr_output.json   # OCR 结果（含 tables[].bbox，作去重标记）
    └── _tmp_table_crops/            # 临时表格图（定期清理）
```

**作业提交模板（A100 / A800）**

```bash
#!/bin/bash
#SBATCH -p A100            # 或 A800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH -o ocr_%j.out

module load apps/anaconda3/2021.05
source activate "$HOME/envs/chandra"
cd "$HOME/esg-pipeline" || exit 1

export MODEL_CHECKPOINT="$HOME/models/chandra-ocr-2"
export HF_HUB_OFFLINE=1
export PDF_INPUT_DIR="HKEX ESG Reports"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OCR_BATCH_SIZE=16

mkdir -p logs
JOB_TAG="${SLURM_JOB_ID:-local}"
python -m src.main --step numeric_extraction --pdf_list_file list_a --force \
  > "logs/ocr_list_a_job${JOB_TAG}.log" 2>&1
```

**日志分析**

```bash
# 异常 PDF：单页 >5 分钟
grep "OCR SUMMARY" logs/*.log | awk -F'|' '{split($1,a," ");split(a[6],b,"s");n=split($2,c,"/");if(b[1]/n>300)print}'
grep -i "cuda error" logs/*.log     # CUDA 错误
bash compare_gpu_speed.sh <job1> <job2>   # 多 job 性能对比
```

---

## 8. 特殊问题 PDF

| PDF | 现象 | 处理 |
|-----|------|------|
| `ltn201707281159.pdf`（01622 REDCO） | 触发 GPU 0 崩溃（Job 1013） | 入黑名单 |
| `2023042100201.pdf`（02382 舜宇） | A800 上 1 页 725s、17 tok/s（正常 162），2598×3484 高分辨率 | 入黑名单 |

异常检测：单页处理 >300s 或 tok/s <50 的 PDF 应排查（见 §7 命令）。

---

## 9. 快速参考

**状态文件**：`/tmp/gpu_health_status.json`、`/tmp/pdf_blacklist.json`
**缓存目录**：`quantitative_results_ocr/chandra_ocr_2/`
**常用命令**：

```bash
sbatch run_ocr.sh          # 提交
scancel <job_id>           # 停止
squeue -u $USER           # 状态
nvidia-smi                 # GPU
grep -c "OCR SUMMARY" logs/ocr_list_*_job*.log | awk -F: '{s+=$2}END{print s}'  # 进度
```

## 10. 运行环境与版本（2026-08-01 实测）

| 项 | 值 | 备注 |
|---|---|---|
| 登录节点默认 `python` | **2.7.5**（`/usr/lib64/python2.7`） | login1 **没有 `python3`**；直接在登录节点跑脚本必须 py2.7 兼容：不能用 f-string；含中文的字符串模板用 `u"..."`；写 JSON 用 `json.dumps(..., ensure_ascii=True)` + 普通文件写入（py2 的 `json.dump` 写 utf-8 流会报 `TypeError: must be unicode, not str`） |

> 编译器版本（GCC / CUDA nvcc）与 job 内实际解释器（管线需 python3，应为 conda/env 激活）待补充。

---

**文档版本**：2.0（精简版） ｜ **最后更新**：2026-08-01
