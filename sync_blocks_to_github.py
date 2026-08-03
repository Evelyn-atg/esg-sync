#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群端：扫描 OCR 结果目录，把 numeric_blocks.json 上传到 GitHub esg-sync/numeric_extracts/{pid}/。

llm_matching 的唯一输入是 numeric_extracts/{pid}/numeric_blocks.json（含全部文本块 + 表格 OCR 块），
所以集群只需上传这个文件即可，不需要 PDF / OCR 缓存。

用法（在集群上）：
    python sync_blocks_to_github.py                          # 默认扫 ./numeric_extracts
    python sync_blocks_to_github.py --src "$HOME/esg-pipeline/numeric_extracts"
    python sync_blocks_to_github.py --dry-run                # 只看会传哪些

依赖：gh CLI 已登录（gh auth status），或设置 GITHUB_TOKEN 环境变量。
幂等：已上传的 pid 自动跳过（断点续传），重复运行安全。
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time

REPO = "Evelyn-atg/esg-sync"
REMOTE_DIR = "numeric_extracts"


def gh(args, _input=None):
    """包装 gh api 调用，返回 (returncode, stdout, stderr)。"""
    cmd = ["gh", "api"]
    if args:
        cmd += args
    if _input is not None:
        cmd += ["--input", "-"]
        r = subprocess.run(cmd, input=_input, capture_output=True, text=True)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def remote_exists(path):
    rc, _, _ = gh(["repos", REPO, "contents", path])
    return rc == 0


def upload(path, content: bytes, msg):
    payload = json.dumps({"message": msg, "content": base64.b64encode(content).decode()})
    rc, _, err = gh(["-X", "PUT", "repos", REPO, "contents", path], _input=payload)
    return rc == 0, err


def main():
    ap = argparse.ArgumentParser(description="扫描集群 OCR 结果目录并上传 numeric_blocks.json 到 GitHub")
    ap.add_argument("--src", default="numeric_extracts", help="集群 OCR 结果目录（含 {pid}/numeric_blocks.json）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将上传的 pid，不实际上传")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    if not os.path.isdir(src):
        print(f"[错误] 源目录不存在: {src}")
        sys.exit(1)

    # 扫描所有含 numeric_blocks.json 的 pid
    candidates = []
    for pid in sorted(os.listdir(src)):
        d = os.path.join(src, pid)
        if not os.path.isdir(d):
            continue
        blocks = os.path.join(d, "numeric_blocks.json")
        if os.path.isfile(blocks):
            size = os.path.getsize(blocks)
            candidates.append((pid, blocks, size))
    print(f"扫描 {src}: 找到 {len(candidates)} 个含 numeric_blocks.json 的报告")

    new_cnt = skip_cnt = fail_cnt = 0
    for pid, blocks, size in candidates:
        remote_path = f"{REMOTE_DIR}/{pid}/numeric_blocks.json"
        if remote_exists(remote_path):
            skip_cnt += 1
            continue
        if args.dry_run:
            print(f"  [dry-run] 将上传 {pid} ({size//1024} KB)")
            new_cnt += 1
            continue
        with open(blocks, "rb") as f:
            content = f.read()
        ok, err = upload(remote_path, content, f"sync numeric_blocks for {pid} from cluster")
        if ok:
            new_cnt += 1
            print(f"  [OK] {pid} ({len(content)//1024} KB)")
        else:
            fail_cnt += 1
            print(f"  [FAIL] {pid}: {err.strip()[:120]}")
        time.sleep(0.3)

    print(f"\n完成: 新增 {new_cnt} / 已存在跳过 {skip_cnt} / 失败 {fail_cnt}")
    if fail_cnt:
        print("有失败项，重跑本脚本即可续传（幂等）")
        sys.exit(1)


if __name__ == "__main__":
    main()
