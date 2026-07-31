---
name: gpu-error-handling-pdf-blacklist
description: GPU 错误处理和 PDF 黑名单机制，防止故障 GPU 继续处理，自动跳过问题 PDF
metadata: 
  node_type: memory
  type: project
  related: "[[gpu-batching-optimization]], [[chandra-gpu-cluster-setup]]"
  originSessionId: 22294c7b-2e97-4645-a148-1a37910e37ce
  modified: 2026-07-31T09:12:30.671Z
---

# GPU 错误处理和 PDF 黑名单机制

## 问题背景

在运行 Chandra OCR 批处理任务时，GPU 0 在运行 5 小时 23 分钟后崩溃：

```
CUDA error: unspecified launch failure
CUDA kernel errors might be asynchronously reported at some other API call
```

**崩溃后果：**
- Job 1013 继续运行了 5+ 小时，但所有 OCR 任务都失败
- 回退到 text_fallback，浪费了约 7400 个 PDF 的处理时间
- GPU 0 显示 "GPU requires reset"，无法继续使用

## 解决方案

### 1. GPU 健康追踪系统（GPUHealthTracker）

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

### 2. PDF 黑名单机制（PDFBlacklist）

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

### 3. 错误处理流程

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

### 4. 黑名单检查

在处理 PDF 前检查黑名单：

```python
# Check if this PDF is blacklisted
pdf_blacklist = PDFBlacklist()
if pdf_blacklist.is_blacklisted(pdf_name):
    logger.warning(f"PDF {pdf_name} is blacklisted. Skipping.")
    return {}  # Skip this PDF
```

## 管理命令

### 查看 GPU 健康状态

```bash
cat /tmp/gpu_health_status.json | python -m json.tool
```

### 查看 PDF 黑名单

```bash
cat /tmp/pdf_blacklist.json | python -m json.tool
```

### 重置 GPU 健康状态（GPU 修复后）

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

### 从黑名单移除 PDF

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

## 特殊问题 PDF

### ltn201707281159

- **股票**：01622 REDCO PPT
- **标题**：ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2016
- **问题**：触发 GPU 0 崩溃
- **特征**：5 页，160KB，无明显异常（肉眼检查）
- **可能原因**：特殊的 PDF 结构或图像编码

### 2023042100201

- **股票**：02382 SUNNY OPTICAL
- **标题**：ENVIRONMENTAL, SOCIAL AND GOVERNANCE REPORT 2022
- **问题**：在 A800 上处理极慢（1 页花 12 分钟）
- **特征**：91 页，12MB，大量 "Invalid Font Weight" 警告
- **图像**：高分辨率（2598×3484 像素），JPEG + JPX 编码
- **可能原因**：复杂图像编码或字体问题导致 GPU 计算瓶颈

## 效果

**优化前：**
- GPU 崩溃后继续运行 5+ 小时，所有 OCR 失败
- 浪费大量计算时间

**优化后：**
- GPU 崩溃后立即停止（几秒钟内）
- 自动标记不健康 GPU
- 问题 PDF 自动加入黑名单
- 其他 worker 可以避免使用故障 GPU

## 相关文件

- `src/utils.py`: GPUHealthTracker, PDFBlacklist, GPUFatalError
- `src/chandra_ocr_tester.py`: CUDA 错误捕获
- `src/numeric_extractor.py`: GPUFatalError 处理，黑名单检查
- `/tmp/gpu_health_status.json`: GPU 健康状态
- `/tmp/pdf_blacklist.json`: PDF 黑名单
