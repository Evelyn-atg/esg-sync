# GPU Fatal Error Handling Implementation

## Overview

This document describes the GPU fatal error handling mechanism implemented to prevent workers from continuing execution after a CUDA error occurs.

## Problem Statement

Previously, when a GPU encountered a CUDA error (e.g., `CUDA error: unspecified launch failure`), the worker would:
1. Log the error
2. Continue processing subsequent batches
3. All subsequent OCR operations would fail
4. Waste significant compute time (in one case, 5+ hours)

## Solution

Implemented a three-layer GPU error handling system:

### 1. GPU Health Tracking (`src/utils.py`)

**Components:**
- `GPUHealthTracker` class: Tracks GPU health status across multiple processes
- `GPUFatalError` exception: Special exception for GPU fatal errors
- Shared JSON file (`/tmp/gpu_health_status.json`): Stores unhealthy GPU list and failure logs

**Key Methods:**
```python
tracker = GPUHealthTracker()
tracker.is_gpu_healthy(gpu_id)              # Check if GPU is healthy
tracker.mark_gpu_unhealthy(gpu_id, error)   # Mark GPU as unhealthy
tracker.get_healthy_gpus(total_gpus)        # Get list of healthy GPUs
tracker.reset_gpu_health(gpu_id)            # Reset GPU health (for recovery)
```

**Features:**
- File locking with `fcntl` for multi-process safety
- Automatic failure logging with timestamp, PID, and error message
- Cross-process GPU health sharing

### 2. CUDA Error Detection (`src/chandra_ocr_tester.py`)

**Modified Method:** `_run_page_ocr_batch_chandra_local()`

**Error Handling Flow:**
```python
try:
    outputs = manager.generate(items)
except RuntimeError as e:
    if "CUDA" in str(e):
        gpu_id = torch.cuda.current_device()
        tracker.mark_gpu_unhealthy(gpu_id, error_msg)
        raise GPUFatalError(...) from e
    else:
        raise
```

**Two-Level Detection:**
1. **Immediate errors**: Caught during `manager.generate()` call
2. **Deferred errors**: Caught during `torch.cuda.synchronize()` after generation

### 3. Worker Exit (`src/numeric_extractor.py`)

**Modified Method:** `_ocr_table_pages()`

**Error Handling:**
```python
try:
    batch_results = ocr_tester._run_page_ocr_batch(batch_images, timing=timing)
except GPUFatalError as e:
    sampler.stop()
    logger.critical(f"FATAL GPU ERROR: {e}")
    logger.critical("Worker exiting immediately. GPU marked as unhealthy.")
    raise  # Re-raise to trigger immediate worker exit
except Exception as e:
    # Handle other errors (continue processing)
    continue
```

## Behavior

### When CUDA Error Occurs:

1. **Detection**: CUDA error caught in `_run_page_ocr_batch_chandra_local()`
2. **Marking**: GPU marked as unhealthy in shared JSON file
3. **Logging**: Critical error logged with full context
4. **Exit**: `GPUFatalError` raised and propagated to worker exit
5. **Worker Termination**: Worker process exits immediately (non-zero exit code)

### What Does NOT Happen:

- ❌ Worker does NOT continue processing
- ❌ Worker does NOT fall back to text extraction
- ❌ Worker does NOT waste time on broken GPU
- ❌ Other workers can check GPU health and avoid the broken GPU

## Usage

### For Workers:

Workers automatically benefit from GPU health tracking:
```python
# At worker startup
tracker = GPUHealthTracker()
gpu_id = torch.cuda.current_device()
if not tracker.is_gpu_healthy(gpu_id):
    logger.error(f"GPU {gpu_id} is unhealthy, exiting")
    sys.exit(1)
```

### For Recovery:

After GPU reset by administrator:
```python
tracker = GPUHealthTracker()
tracker.reset_gpu_health(gpu_id)
```

### Monitoring:

Check GPU health status:
```bash
cat /tmp/gpu_health_status.json
```

Example output:
```json
{
  "unhealthy_gpus": [0],
  "failure_log": [
    {
      "gpu_id": 0,
      "timestamp": "2024-07-30T07:03:05.839123",
      "error": "CUDA error: unspecified launch failure",
      "pid": 12345
    }
  ]
}
```

## Testing

To test the error handling:
1. Manually trigger a CUDA error (e.g., allocate too much memory)
2. Observe worker exits immediately
3. Check `/tmp/gpu_health_status.json` for GPU marked unhealthy
4. Restart worker - should refuse to use the unhealthy GPU

## Future Improvements

1. **Automatic GPU switching**: Worker could automatically switch to a healthy GPU
2. **Health check integration**: Pre-flight GPU health check before starting work
3. **Alerting**: Send alerts when GPU marked unhealthy
4. **Metrics**: Track GPU failure rate over time
5. **Recovery automation**: Automatically reset GPU health after admin confirms fix

## Files Modified

- `src/utils.py`: Added `GPUHealthTracker` and `GPUFatalError`
- `src/chandra_ocr_tester.py`: Added CUDA error detection in batch processing
- `src/numeric_extractor.py`: Added `GPUFatalError` handling with immediate exit

## Compatibility

- Works with multi-process workers (file locking ensures safety)
- Compatible with both HF and vLLM backends
- No performance impact on normal operation
- Minimal overhead (JSON file I/O only on GPU failure)
