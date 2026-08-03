#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集群端：扫描 OCR 结果目录，把 numeric_blocks.json 上传到 GitHub esg-pipeline/numeric_extracts/{pid}/。

不依赖 gh CLI——用 requests + GITHUB_TOKEN 直连 GitHub REST API。
llm_matching 的唯一输入是 numeric_extracts/{pid}/numeric_blocks.json（含全部文本块 + 表格 OCR 块），
所以集群只需上传这个文件即可，不需要 PDF / OCR 缓存。

用法（在集群上，chandra 环境）：
    GITHUB_TOKEN=xxx python sync_blocks_to_github.py --list-file /tmp/todo65.txt --dry-run   # 预览
    GITHUB_TOKEN=xxx python sync_blocks_to_github.py --list-file /tmp/todo65.txt             # 正式传
    python sync_blocks_to_github.py --token xxx --list-file /tmp/todo65.txt                  # 或用 --token

幂等：已上传的 pid 自动跳过（断点续传），重复运行安全。
"""
import argparse
import base64
import json
import os
import sys
import time

import requests

REPO = "Evelyn-atg/esg-pipeline"
REMOTE_DIR = "numeric_extracts"
API = "https://api.github.com"


def _headers(token):
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def remote_exists(token, path):
    r = requests.get(f"{API}/repos/{REPO}/contents/{path}", headers=_headers(token), timeout=30)
    return r.status_code == 200


def upload(token, path, content, msg):
    payload = {"message": msg, "content": base64.b64encode(content).decode()}
    r = requests.put(f"{API}/repos/{REPO}/contents/{path}", headers=_headers(token), json=payload, timeout=60)
    return r.status_code in (200, 201), r.text[:150]


def main():
    ap = argparse.ArgumentParser(description="扫描集群 OCR 结果目录并上传 numeric_blocks.json 到 GitHub（requests 版，无需 gh CLI）")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                    help="GitHub token（有 repo 写权限；或用 GITHUB_TOKEN 环境变量）")
    ap.add_argument("--src", default="numeric_extracts", help="集群 OCR 结果目录（含 {pid}/numeric_blocks.json）")
    ap.add_argument("--list-file", default=None, help="只上传此清单中的 pid（每行一个，支持 pid 或 pid.pdf）")
    ap.add_argument("--limit", type=int, default=0, help="最多上传 N 个（0=全部；配合 --list-file 取清单前 N 个）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将上传的 pid，不实际上传")
    args = ap.parse_args()

    if not args.token:
        print("[错误] 需要 GITHUB_TOKEN（环境变量）或 --token 参数。")
        print("       生成方式: GitHub → Settings → Developer settings → Personal access tokens → Generate")
        sys.exit(1)

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
            candidates.append((pid, blocks, os.path.getsize(blocks)))
    print(f"扫描 {src}: 找到 {len(candidates)} 个含 numeric_blocks.json 的报告")

    # 清单过滤：只传用户指定的 pid（防止 6500 个全传爆 GitHub）
    if args.list_file:
        want = set()
        for line in open(args.list_file, encoding="utf-8"):
            pid = line.strip().replace(".pdf", "").replace("/", "").replace(".", "")
            if pid:
                want.add(pid)
        candidates = [c for c in candidates if c[0] in want]
        print(f"清单过滤（{args.list_file}）: 命中 {len(candidates)} 个")
    if args.limit > 0:
        candidates = candidates[:args.limit]
        print(f"limit={args.limit}: 只传前 {len(candidates)} 个")

    new_cnt = skip_cnt = fail_cnt = 0
    for pid, blocks, size in candidates:
        remote_path = f"{REMOTE_DIR}/{pid}/numeric_blocks.json"
        if remote_exists(args.token, remote_path):
            skip_cnt += 1
            continue
        if args.dry_run:
            print(f"  [dry-run] 将上传 {pid} ({size//1024} KB)")
            new_cnt += 1
            continue
        with open(blocks, "rb") as f:
            content = f.read()
        ok, err = upload(args.token, remote_path, content, f"sync numeric_blocks for {pid} from cluster")
        if ok:
            new_cnt += 1
            print(f"  [OK] {pid} ({len(content)//1024} KB)")
        else:
            fail_cnt += 1
            print(f"  [FAIL] {pid}: {err[:120]}")
        time.sleep(0.3)

    print(f"\n完成: 新增 {new_cnt} / 已存在跳过 {skip_cnt} / 失败 {fail_cnt}")
    if fail_cnt:
        print("有失败项，重跑本脚本即可续传（幂等）")
        sys.exit(1)


if __name__ == "__main__":
    main()
