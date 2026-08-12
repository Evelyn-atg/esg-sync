#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESG OCR 输出文件批量上传到阿里云 OSS（深圳节点）

扫描集群上的 OCR 输出和数值提取结果，批量上传到阿里云 OSS。
支持断点续传、多线程并行、自动重试、进度日志、校验。

用法:
  python3 upload_ocr_to_oss.py                     # 全量上传
  python3 upload_ocr_to_oss.py --resume            # 断点续传（跳过已上传）
  python3 upload_ocr_to_oss.py --dry-run           # 只扫描不上传
  python3 upload_ocr_to_oss.py --workers 16        # 16 线程并行
  python3 upload_ocr_to_oss.py --only ocr          # 只上传 OCR 输出
  python3 upload_ocr_to_oss.py --only extract      # 只上传数值提取
  python3 upload_ocr_to_oss.py --verify            # 校验 OSS 上的文件
  python3 upload_ocr_to_oss.py --list              # 列出 OSS 上的文件

OSS 路径结构:
  esg-ocr/chandra_ocr_2/{pid}/{pid}_ocr_output.json
  esg-ocr/numeric_extracts/{pid}/numeric_blocks.json
  esg-ocr/upload_manifest.json
"""

import os
import sys
import time
import json
import shutil
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ==================== .env 加载 ====================

def load_env():
    """从同目录 .env 文件加载环境变量"""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    # 优先用 python-dotenv（如果装了）
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        pass

load_env()

import oss2

# ==================== 配置 ====================

DEFAULT_BASE_DIR = os.environ.get(
    "ESG_BASE_DIR", "/public/home/zixin/esg-pipeline"
)
DEFAULT_OSS_PREFIX = "esg-ocr/"
DEFAULT_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, multiplied by attempt number

# ==================== OSS 初始化 ====================


def get_bucket():
    """初始化 OSS Bucket，返回 (bucket, bucket_name, endpoint)"""
    access_id = os.getenv("OSS_ACCESS_KEY_ID")
    access_key = os.getenv("OSS_ACCESS_KEY_SECRET")
    endpoint = os.getenv("OSS_ENDPOINT")
    bucket_name = os.getenv("OSS_BUCKET_NAME")

    if not all([access_id, access_key, endpoint, bucket_name]):
        print("[ERROR] OSS config missing. Check .env file.")
        print(f"  OSS_ACCESS_KEY_ID:     {'set' if access_id else 'MISSING'}")
        print(f"  OSS_ACCESS_KEY_SECRET: {'set' if access_key else 'MISSING'}")
        print(f"  OSS_ENDPOINT:          {endpoint or 'MISSING'}")
        print(f"  OSS_BUCKET_NAME:       {bucket_name or 'MISSING'}")
        sys.exit(1)

    auth = oss2.Auth(access_id, access_key)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return bucket, bucket_name, endpoint


# ==================== 扫描 ====================


def scan_ocr(base_dir, oss_prefix):
    """扫描 chandra_ocr_2 目录下所有 {pid}_ocr_output.json"""
    ocr_dir = Path(base_dir) / "quantitative_results_ocr" / "chandra_ocr_2"
    files = []
    if not ocr_dir.exists():
        print(f"[WARN] OCR dir not found: {ocr_dir}")
        return files

    for d in sorted(ocr_dir.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        jf = d / f"{pid}_ocr_output.json"
        if jf.exists():
            files.append(
                {
                    "local_path": str(jf),
                    "oss_key": f"{oss_prefix}chandra_ocr_2/{pid}/{pid}_ocr_output.json",
                    "pid": pid,
                    "type": "ocr",
                    "size": jf.stat().st_size,
                }
            )
    return files


def scan_extracts(base_dir, oss_prefix):
    """扫描 numeric_extracts 目录下所有 numeric_blocks.json"""
    ext_dir = Path(base_dir) / "numeric_extracts"
    files = []
    if not ext_dir.exists():
        print(f"[WARN] Extracts dir not found: {ext_dir}")
        return files

    for d in sorted(ext_dir.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        jf = d / "numeric_blocks.json"
        if jf.exists():
            files.append(
                {
                    "local_path": str(jf),
                    "oss_key": f"{oss_prefix}numeric_extracts/{pid}/numeric_blocks.json",
                    "pid": pid,
                    "type": "extract",
                    "size": jf.stat().st_size,
                }
            )
    return files


# ==================== 上传 ====================


def upload_one(bucket, file_info):
    """上传单个文件，带重试"""
    local_path = file_info["local_path"]
    oss_key = file_info["oss_key"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = bucket.put_object_from_file(oss_key, local_path)
            return {
                "oss_key": oss_key,
                "pid": file_info["pid"],
                "type": file_info["type"],
                "size": file_info["size"],
                "etag": result.etag,
                "status": "ok",
                "attempts": attempt,
            }
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                return {
                    "oss_key": oss_key,
                    "pid": file_info["pid"],
                    "type": file_info["type"],
                    "size": file_info["size"],
                    "status": "error",
                    "error": str(e),
                    "attempts": attempt,
                }


def get_existing_keys(bucket, prefix):
    """列出 OSS 上已存在的对象 key（用于断点续传）"""
    existing = set()
    print(f'[INFO] Listing existing OSS objects under "{prefix}" ...')
    count = 0
    for obj in oss2.ObjectIterator(bucket, prefix=prefix, max_keys=1000):
        existing.add(obj.key)
        count += 1
        if count % 2000 == 0:
            print(f"  ... {count} objects listed")
    print(f"[INFO] Found {len(existing)} existing objects on OSS")
    return existing


# ==================== 校验 ====================


def verify_files(bucket, all_files):
    """校验 OSS 上的文件是否存在且大小匹配"""
    print(f"[VERIFY] Checking {len(all_files)} files on OSS ...")
    ok = 0
    missing = 0
    size_mismatch = 0

    for i, f in enumerate(all_files, 1):
        exists = bucket.object_exists(f["oss_key"])
        if not exists:
            missing += 1
            print(f"  [MISSING] {f['oss_key']}")
        else:
            meta = bucket.head_object(f["oss_key"])
            oss_size = meta.content_length
            if oss_size != f["size"]:
                size_mismatch += 1
                print(
                    f"  [SIZE MISMATCH] {f['oss_key']}: "
                    f"local={f['size']} oss={oss_size}"
                )
            else:
                ok += 1

        if i % 500 == 0:
            print(f"  ... checked {i}/{len(all_files)}")

    print(f"\n[VERIFY] OK: {ok} | Missing: {missing} | Size mismatch: {size_mismatch}")
    return missing == 0 and size_mismatch == 0


# ==================== 主逻辑 ====================


def main():
    parser = argparse.ArgumentParser(
        description="ESG OCR output -> Alibaba Cloud OSS uploader"
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Project root dir (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files already on OSS (checkpoint resume)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel upload threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan only, no upload"
    )
    parser.add_argument(
        "--only",
        choices=["ocr", "extract"],
        help="Upload only OCR output or only numeric extracts",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify uploaded files on OSS"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all files on OSS under prefix"
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_OSS_PREFIX,
        help=f"OSS key prefix (default: {DEFAULT_OSS_PREFIX})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ESG OCR -> Alibaba Cloud OSS Uploader")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Init OSS
    bucket, bucket_name, endpoint = get_bucket()
    print(f"[INFO] Bucket:   {bucket_name}")
    print(f"[INFO] Endpoint: {endpoint}")
    print(f"[INFO] Base dir: {args.base_dir}")
    print(f"[INFO] Prefix:   {args.prefix}")
    print()

    # --- List mode ---
    if args.list:
        print("[LIST] Objects on OSS:")
        count = 0
        for obj in oss2.ObjectIterator(bucket, prefix=args.prefix, max_keys=10000):
            count += 1
            if count <= 20:
                print(f"  {obj.key}  ({obj.size} bytes)")
            elif count % 1000 == 0:
                print(f"  ... {count} objects")
        print(f"\n[LIST] Total: {count} objects")
        return

    # --- Scan local files ---
    all_files = []
    if args.only != "extract":
        ocr_files = scan_ocr(args.base_dir, args.prefix)
        print(f"[SCAN] OCR output files:    {len(ocr_files)}")
        all_files.extend(ocr_files)
    if args.only != "ocr":
        ext_files = scan_extracts(args.base_dir, args.prefix)
        print(f"[SCAN] Numeric extract files: {len(ext_files)}")
        all_files.extend(ext_files)

    total_size = sum(f["size"] for f in all_files)
    print(f"[SCAN] Total files:         {len(all_files)}")
    print(f"[SCAN] Total size:          {total_size / 1024 / 1024:.1f} MB")
    print()

    # --- Dry run ---
    if args.dry_run:
        print("[DRY RUN] No files uploaded. Sample:")
        for f in all_files[:5]:
            print(
                f"  {f['type']:8s} {f['pid']} -> {f['oss_key']} "
                f"({f['size'] / 1024:.1f} KB)"
            )
        if len(all_files) > 5:
            print(f"  ... and {len(all_files) - 5} more")
        return

    if not all_files:
        print("[INFO] No files to upload.")
        return

    # --- Verify mode ---
    if args.verify:
        verify_files(bucket, all_files)
        return

    # --- Resume: filter out already uploaded ---
    if args.resume:
        existing = get_existing_keys(bucket, args.prefix)
        before = len(all_files)
        all_files = [f for f in all_files if f["oss_key"] not in existing]
        skipped = before - len(all_files)
        print(f"[RESUME] Skipping {skipped} already uploaded files")
        print(f"[RESUME] {len(all_files)} files remaining to upload")

    if not all_files:
        print("[INFO] All files already on OSS. Nothing to do.")
        return

    # --- Upload ---
    print(
        f"\n[UPLOAD] Starting upload of {len(all_files)} files "
        f"with {args.workers} threads ...\n"
    )

    results = []
    ok_count = 0
    err_count = 0
    uploaded_bytes = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(upload_one, bucket, f): f for f in all_files
        }
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            if result["status"] == "ok":
                ok_count += 1
                uploaded_bytes += result["size"]
            else:
                err_count += 1
                print(
                    f"  [ERROR] {result['pid']} ({result['type']}): "
                    f"{result.get('error', 'unknown')}"
                )

            if done % 100 == 0 or done == total:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{total}] ok={ok_count} err={err_count} | "
                    f"{rate:.1f} files/s | "
                    f"{uploaded_bytes / 1024 / 1024:.1f} MB sent | "
                    f"ETA {eta:.0f}s"
                )

    elapsed = time.time() - start_time

    # --- Summary ---
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files:    {len(all_files)}")
    print(f"Successful:     {ok_count}")
    print(f"Failed:         {err_count}")
    print(f"Time elapsed:   {elapsed:.1f}s")
    print(f"Data uploaded:  {uploaded_bytes / 1024 / 1024:.1f} MB")
    if elapsed > 0:
        print(f"Average speed:  {uploaded_bytes / 1024 / 1024 / elapsed:.1f} MB/s")

    # --- Save manifest ---
    script_dir = Path(__file__).resolve().parent
    manifest_path = script_dir / "upload_manifest.json"
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "base_dir": args.base_dir,
        "oss_prefix": args.prefix,
        "bucket": bucket_name,
        "endpoint": endpoint,
        "total_files": len(all_files),
        "successful": ok_count,
        "failed": err_count,
        "elapsed_seconds": round(elapsed, 1),
        "data_uploaded_mb": round(uploaded_bytes / 1024 / 1024, 1),
        "files": sorted(results, key=lambda x: (x["type"], x["pid"])),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Manifest saved: {manifest_path}")

    # Upload manifest to OSS
    try:
        bucket.put_object_from_file(
            f"{args.prefix}upload_manifest.json", str(manifest_path)
        )
        print(
            f"[INFO] Manifest uploaded to OSS: "
            f"{args.prefix}upload_manifest.json"
        )
    except Exception as e:
        print(f"[WARN] Failed to upload manifest: {e}")

    # --- Save error log ---
    if err_count > 0:
        err_path = script_dir / "upload_errors.json"
        errors = [r for r in results if r["status"] == "error"]
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Error log saved: {err_path}")
        print("[INFO] Re-run with --resume to retry failed files.")

    print()
    if err_count == 0:
        print("All files uploaded successfully!")
    else:
        print(f"{err_count} files failed. Check upload_errors.json and re-run with --resume.")


if __name__ == "__main__":
    main()
