import logging
from pathlib import Path
import json
from typing import List, Dict, Any

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers to the same logger
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def ensure_directory_exists(path: Path) -> None:
    """
    Ensure that the specified directory exists
    """
    path.mkdir(parents=True, exist_ok=True)

def save_json(data: Any, filepath: Path) -> None:
    """
    Save data to a JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: Path) -> Any:
    """
    Load data from a JSON file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_pdf_files(pdf_dir: Path, limit: int = None) -> List[Path]:
    """
    Get all PDF files from the specified directory, optionally limiting the count
    """
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if limit:
        return pdf_files[:limit]
    return pdf_files

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be filesystem-friendly
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


# ===== GPU Health Tracking =====
import os
import fcntl
from datetime import datetime

_GPU_HEALTH_FILE = Path("/tmp/gpu_health_status.json")


class GPUHealthTracker:
    """Track GPU health status across multiple processes.

    Stores health status in a shared JSON file so all workers can check
    which GPUs are unhealthy and avoid using them.
    """

    def __init__(self):
        self.health_file = _GPU_HEALTH_FILE

    def _load_health_data(self) -> Dict[str, Any]:
        """Load GPU health data from shared file with file locking."""
        if not self.health_file.exists():
            return {"unhealthy_gpus": [], "failure_log": []}

        try:
            with open(self.health_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return data
        except Exception:
            return {"unhealthy_gpus": [], "failure_log": []}

    def _save_health_data(self, data: Dict[str, Any]) -> None:
        """Save GPU health data to shared file with file locking."""
        try:
            with open(self.health_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2, ensure_ascii=False)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logging.error(f"Failed to save GPU health data: {e}")

    def is_gpu_healthy(self, gpu_id: int) -> bool:
        """Check if a GPU is healthy."""
        data = self._load_health_data()
        return gpu_id not in data.get("unhealthy_gpus", [])

    def mark_gpu_unhealthy(self, gpu_id: int, error_msg: str) -> None:
        """Mark a GPU as unhealthy and log the failure."""
        data = self._load_health_data()

        if gpu_id not in data.get("unhealthy_gpus", []):
            data.setdefault("unhealthy_gpus", []).append(gpu_id)

        # Log the failure
        data.setdefault("failure_log", []).append({
            "gpu_id": gpu_id,
            "timestamp": datetime.now().isoformat(),
            "error": error_msg,
            "pid": os.getpid()
        })

        self._save_health_data(data)
        logging.error(f"GPU {gpu_id} marked as unhealthy: {error_msg}")

    def get_healthy_gpus(self, total_gpus: int) -> List[int]:
        """Get list of healthy GPU IDs."""
        data = self._load_health_data()
        unhealthy = set(data.get("unhealthy_gpus", []))
        return [i for i in range(total_gpus) if i not in unhealthy]

    def reset_gpu_health(self, gpu_id: int) -> None:
        """Reset a GPU's health status (for testing/recovery)."""
        data = self._load_health_data()
        if gpu_id in data.get("unhealthy_gpus", []):
            data["unhealthy_gpus"].remove(gpu_id)
            self._save_health_data(data)
            logging.info(f"GPU {gpu_id} health status reset")


class GPUFatalError(Exception):
    """Exception raised when a fatal GPU error is detected."""
    pass
