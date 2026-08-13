#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 chandra_ocr_2 和 numeric_extracts 的 PID 差异，找出需要补跑 OCR 的 PDF。

背景：
  - OCR 输出:   quantitative_results_ocr/chandra_ocr_2/{pid}/{pid}_ocr_output.json
  - 数值提取:   numeric_extracts/{pid}/numeric_blocks.json
  - 现象: numeric_extracts(9636) > OCR(9391)，说明有 PID 有 numeric 但缺 OCR

判断逻辑：
  numeric_blocks.json 里 block.source 含 table_ocr / table_page → 该 PDF 有表格页，需要 OCR
  source 全为 text → 纯文本 PDF，不需要 OCR

用法（集群，先 module load）:
  python3 check_missing_ocr.py
"""
import os
import json
from pathlib import Path
from collections import Counter

BASE = Path(os.environ.get("ESG_BASE", "/public/home/zixin/esg-pipeline"))
OCR_DIR = BASE / "quantitative_results_ocr" / "chandra_ocr_2"
NUM_DIR = BASE / "numeric_extracts"

# 1. 收集 OCR PID
ocr_pids = set()
if OCR_DIR.exists():
    for d in OCR_DIR.iterdir():
        if d.is_dir() and (d / f"{d.name}_ocr_output.json").exists():
            ocr_pids.add(d.name)

# 2. 收集 numeric PID
num_pids = set()
if NUM_DIR.exists():
    for d in NUM_DIR.iterdir():
        if d.is_dir() and (d / "numeric_blocks.json").exists():
            num_pids.add(d.name)

only_num = num_pids - ocr_pids   # 有 numeric 但缺 OCR
only_ocr = ocr_pids - num_pids   # 有 OCR 但缺 numeric

print("=" * 62)
print("  OCR vs numeric_extracts PID 差异检查")
print("=" * 62)
print(f"  OCR 输出 PID 数:    {len(ocr_pids)}")
print(f"  Numeric PID 数:     {len(num_pids)}")
print(f"  有 numeric 缺 OCR:  {len(only_num)}   ← 重点")
print(f"  有 OCR 缺 numeric:  {len(only_ocr)}")
print("=" * 62)

# 3. 对 only_num 分类
need_ocr = []     # 含表格块，需补跑 OCR
no_table = []     # 纯文本，不需要 OCR
read_err = []     # 读取失败

for pid in sorted(only_num):
    nb = NUM_DIR / pid / "numeric_blocks.json"
    try:
        blocks = json.load(open(nb, encoding="utf-8"))
        if not isinstance(blocks, list):
            blocks = []
        srcs = Counter(b.get("source", "?") for b in blocks if isinstance(b, dict))
        has_table = srcs.get("table_ocr", 0) > 0 or srcs.get("table_page", 0) > 0
        if has_table:
            need_ocr.append((pid, srcs))
        else:
            no_table.append(pid)
    except Exception as e:
        read_err.append((pid, str(e)))

print(f"\n  有 numeric 但缺 OCR 的 {len(only_num)} 个 PID 分类：")
print(f"    🔴 含表格块（需补跑 OCR）: {len(need_ocr)}")
print(f"    🟢 纯文本（不需要 OCR）:   {len(no_table)}")
print(f"    ⚠️  读取失败:              {len(read_err)}")

# 4. 输出需补跑 OCR 的明细
if need_ocr:
    print(f"\n  === 需补跑 OCR 的 PID（{len(need_ocr)} 个）===")
    for pid, srcs in need_ocr:
        print(f"    {pid}  (source: {dict(srcs)})")
    # 生成 pdf_list 文件，供补跑脚本使用
    out = BASE / "need_ocr.txt"
    with open(out, "w") as f:
        for pid, _ in need_ocr:
            f.write(pid + "\n")
    print(f"\n  已生成 {out}（{len(need_ocr)} 个 PID，逗号分隔可直接用）")

# 5. 有 OCR 缺 numeric 的（顺带提示）
if only_ocr:
    print(f"\n  === 有 OCR 但缺 numeric 的 PID（{len(only_ocr)} 个）===")
    for pid in sorted(only_ocr):
        print(f"    {pid}")

if read_err:
    print(f"\n  === 读取失败（{len(read_err)} 个）===")
    for pid, e in read_err:
        print(f"    {pid}: {e}")

print("\n  Done.")
