#!/usr/bin/env python
# -*- coding: utf-8 -*-
# v2: exact matching via table_block_id (found in block 'source'/'content'),
#     table preview via content_html. py2.7-safe (no f-strings, u"" literals,
#     json writes via ensure_ascii=True + plain open).
from __future__ import print_function
import os, sys, io, json, random, glob, shutil, datetime, argparse, re

TBL_ID_RE = re.compile(r"/page/\d+/Table/\d+")


def load(path):
    try:
        return json.load(io.open(path, "r", encoding="utf-8"))
    except Exception as e:
        return {"__error__": u"{}: {}".format(type(e).__name__, e)}


def find_tables(d):
    if not isinstance(d, dict):
        return []
    for k in ("tables", "table_blocks", "ocr_tables"):
        if isinstance(d.get(k), list):
            return d[k]
    if isinstance(d.get("data"), dict) and isinstance(d["data"].get("tables"), list):
        return d["data"]["tables"]
    return []


def pick_blocks(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("blocks", "numeric_blocks", "results", "items"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return []


def page_of(x):
    if isinstance(x, dict):
        for k in ("page", "page_num", "page_no", "page_idx", "pageno", "page_number"):
            if x.get(k) is not None:
                return x[k]
    return None


def table_id(t):
    if isinstance(t, dict):
        v = t.get("table_block_id")
        if v is not None:
            return u"{}".format(v)
    return None


def block_ids(b):
    if not isinstance(b, dict):
        return set()
    out = set()
    for k in ("source", "content", "text", "markdown"):
        v = b.get(k)
        if isinstance(v, dict):
            v = json.dumps(v, ensure_ascii=True)
        if v is not None:
            out |= set(TBL_ID_RE.findall(u"{}".format(v)))
    return out


def has_table_flag(b):
    if isinstance(b, dict):
        return bool(b.get("has_table"))
    return False


def preview(x, n=90):
    if isinstance(x, dict):
        for k in ("content_html", "html", "content", "text", "markdown"):
            if x.get(k):
                return u"{}".format(x[k]).replace(u"\n", u" ").replace(u"<", u"<")[:n]
        s = json.dumps(x, ensure_ascii=True)
        return s[:n]
    return u"{}".format(x)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--ocr-root", default="quantitative_results_ocr/chandra_ocr_2")
    ap.add_argument("--num-root", default="numeric_extracts")
    ap.add_argument("--out", default="spotcheck_report")
    args = ap.parse_args()

    if not os.path.isdir(args.ocr_root):
        sys.exit(u"OCR root not found: {}\n(run this from inside ~/esg-pipeline)".format(args.ocr_root))

    pdf_dirs = sorted(p for p in glob.glob(os.path.join(args.ocr_root, "*")) if os.path.isdir(p))
    if not pdf_dirs:
        sys.exit(u"No PDF dirs under " + args.ocr_root)

    random.seed(args.seed)
    n = min(args.n, len(pdf_dirs))
    chosen = random.sample(pdf_dirs, n)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = os.path.join(args.out, "report_" + ts)
    ocr_dir = os.path.join(rundir, "ocr")
    num_dir = os.path.join(rundir, "numeric")
    for d in (ocr_dir, num_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    md = [u"# OCR 抽样核对报告 v2（按 table_block_id 精确匹配）", u"",
          u"- 时间: " + ts, u"- seed: `{}`".format(args.seed),
          u"- 抽样数: {}".format(n),
          u"- ocr_root: `{}`".format(args.ocr_root),
          u"- num_root: `{}`".format(args.num_root), u""]

    summary_rows = [u"| PDF | OCR表数 | numeric块数 | 块中 has_table | 精确命中 | 真正被排除 |",
                    u"|---|---|---|---|---|---|"]
    manifest = {"generated_at": ts, "seed": args.seed, "n_total": len(pdf_dirs), "n_sampled": n, "samples": []}

    for d in chosen:
        pdf = os.path.basename(d)
        ocr_file = os.path.join(d, pdf + "_ocr_output.json")
        num_file = os.path.join(args.num_root, pdf, "numeric_blocks.json")

        ocr = load(ocr_file)
        tables = find_tables(ocr)
        num = load(num_file)
        blocks = pick_blocks(num)

        if os.path.exists(ocr_file):
            shutil.copy(ocr_file, os.path.join(ocr_dir, pdf + ".json"))
        if os.path.exists(num_file):
            shutil.copy(num_file, os.path.join(num_dir, pdf + ".json"))

        md.append(u"## " + pdf)
        if "__error__" in ocr or "__error__" in num:
            md.append(u"- 读取错误: ocr={} num={}".format(ocr.get("__error__"), num.get("__error__")))
            md.append(u"")
            continue

        n_tables = len(tables)
        n_blocks = len(blocks)
        n_ht = sum(1 for b in blocks if has_table_flag(b))

        ids = {}
        for i, t in enumerate(tables):
            tid = table_id(t)
            if tid:
                ids.setdefault(tid, []).append(i)

        # every table id that appears in any block -> covered (made it into final output)
        covered = set()
        block_src = []
        for i, b in enumerate(blocks):
            bids = block_ids(b)
            covered |= bids
            block_src.append((i, page_of(b), has_table_flag(b), u",".join(sorted(bids)) or u"-",
                              preview(b.get("source") if isinstance(b, dict) else None, 40)))

        excluded = []
        for tid, idxs in sorted(ids.items()):
            if tid not in covered:
                for i in idxs:
                    t = tables[i]
                    excluded.append({"table_block_id": tid, "page": page_of(t),
                                     "preview": preview(t, 80)})

        # blocks that claim has_table but carry no recognizable table id
        orphan_ht = [i for i, b in enumerate(blocks) if has_table_flag(b) and not block_ids(b)]

        md.append(u"- 表格 {} 张 / numeric 块 {} 个（其中 has_table={}）".format(n_tables, n_blocks, n_ht))
        md.append(u"- numeric 块 source 摘要: {}".format(
            u"; ".join(u"B{} page={} has_table={} ids=[{}] src={}".format(i, p, h, ids2, s)
                       for i, p, h, ids2, s in block_src)))
        if excluded:
            md.append(u"- **真正被排除（OCR 检出但未进入任何 numeric 块）: {} 张**".format(len(excluded)))
            for e in excluded:
                md.append(u"  - {} page={} : {}".format(e["table_block_id"], e["page"], e["preview"]))
        else:
            md.append(u"- **真正被排除: 0 张（全部 OCR 表都有对应 numeric 块）**")
        if orphan_ht:
            md.append(u"- 注意: {} 个块 has_table=True 但未带 table_block_id（可能按内容识别，未计入精确命中）".format(len(orphan_ht)))
        md.append(u"")

        summary_rows.append(u"| {} | {} | {} | {} | {} | {} |".format(
            pdf, n_tables, n_blocks, n_ht, len(ids) - len(excluded), len(excluded)))
        manifest["samples"].append({"pdf": pdf, "n_tables": n_tables, "n_blocks": n_blocks,
                                    "n_blocks_has_table": n_ht,
                                    "n_exact_hit": len(ids) - len(excluded),
                                    "n_excluded": len(excluded),
                                    "excluded": excluded})

    md = md[:8] + summary_rows + [u""] + md[8:]

    with open(os.path.join(rundir, "report.json"), "w") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=True))
    io.open(os.path.join(rundir, "report.md"), "w", encoding="utf-8").write(u"\n".join(md))

    print(u"== Summary ==")
    for s in manifest["samples"]:
        print(u"  {}  tables={}  blocks={}  has_table={}  exact_hit={}  EXCLUDED={}".format(
            s["pdf"], s["n_tables"], s["n_blocks"], s["n_blocks_has_table"],
            s["n_exact_hit"], s["n_excluded"]))
    print(u"== Excluded details ==")
    for s in manifest["samples"]:
        if s["excluded"]:
            print(u"  [{}]".format(s["pdf"]))
            for e in s["excluded"]:
                print(u"    {} page={} : {}".format(e["table_block_id"], e["page"], e["preview"]))
    print(u"Report files: " + os.path.abspath(rundir))


if __name__ == "__main__":
    main()
