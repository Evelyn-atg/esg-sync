#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询 OSS bucket 已上传的对象数量/存储量（替代网页控制台看进度）。

用法（集群，需先 module load）:
  python3 query_oss_stat.py
"""
import os
from pathlib import Path

try:
    import oss2
except ImportError:
    os.system("python3 -m pip install oss2 --quiet")
    import oss2

SCRIPT_DIR = Path(__file__).resolve().parent

# 手动读 .env（不依赖 dotenv）
env = {}
env_path = SCRIPT_DIR / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

AK = env.get("OSS_ACCESS_KEY_ID")
SK = env.get("OSS_ACCESS_KEY_SECRET")
EP = env.get("OSS_ENDPOINT")
BK = env.get("OSS_BUCKET_NAME")

if not (AK and SK and EP and BK):
    print("[ERROR] .env 缺配置: OSS_ACCESS_KEY_ID/SECRET/ENDPOINT/BUCKET_NAME")
    raise SystemExit(1)

auth = oss2.Auth(AK, SK)
bucket = oss2.Bucket(auth, EP, BK)

stat = bucket.get_bucket_stat()
total_expected = 19027          # dry-run 扫描到的总数
total_size_mb = 1783.8          # dry-run 扫描到的总大小 (MB)

cnt = stat.object_count
size_mb = stat.storage_size_in_bytes / 1024 / 1024

print("=" * 46)
print("  OSS 上传进度")
print("=" * 46)
print(f"  Bucket:    {BK}")
print(f"  对象数量:  {cnt} / {total_expected}")
print(f"  完成率:    {cnt * 100 / total_expected:.1f}%")
print(f"  存储量:    {size_mb:.1f} MB / {total_size_mb:.1f} MB")
print("=" * 46)
if cnt >= total_expected:
    print("  ✅ 已全部上传完成")
else:
    print(f"  🔵 进行中（还差 {total_expected - cnt} 个）")
