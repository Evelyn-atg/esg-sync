#!/usr/bin/env python
# -*- coding: utf-8 -*-
# py2.7-safe: no f-strings, unicode literals for any string that may carry CJK,
# JSON writes via json.dumps(ensure_ascii=True) + plain open().
from __future__ import print_function
import os, sys, io, json, random, glob, shutil, datetime, argparse


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


def preview(v, n=100):
    if isinstance(v, dict):
        for k in ("html", "text", "markdown", "content"):
            if v.get(k):
                return u"{}".format(v[k]).replace(u"\n", u" ")[:n]
        s = json.dumps(v, ensure_ascii=True)
        return s[:n]
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=True)[:n]
    return u"{}".format(v)[:n]


def quality(t):
    if isinstance(t, dict):
        q = t.get("parse_quality_score")
        if q is not None:
            try:
                return float(q)
            except Exception:
                pass
    return None


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

    md = [u"# OCR 抽样核对报告（被排除表格分析）", u"",
          u"- 时间: " + ts, u"- seed: `{}`".format(args.seed),
          u"- 抽样数: {}".format(n), u"- ocr_root: `{}`".format(args.ocr_root),
          u"- num_root: `{}`".format(args.num_root), u""]

    summary_rows = [u"| PDF | OCR表数 | numeric块数 | 疑似被排除表 | 说明 |",
                    u"|---|---|---|---|---|"]
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
        if "__error__" in ocr:
            md.append(u"- OCR 读取失败: " + ocr["__error__"])
            md.append(u"")
            continue
        if "__error__" in num:
            md.append(u"- numeric 读取失败: " + num["__error__"])
            md.append(u"")

        n_tables = len(tables)
        n_blocks = len(blocks)

        # schema sample (first table / first block keys) - helps refine matching
        t_keys = u", ".join(u"{}".format(k) for k in (tables[0].keys() if tables and isinstance(tables[0], dict) else []))
        b_keys = u", ".join(u"{}".format(k) for k in (blocks[0].keys() if blocks and isinstance(blocks[0], dict) else []))
        md.append(u"- OCR 表字段: " + (t_keys or u"(无)"))
        md.append(u"- numeric 块字段: " + (b_keys or u"(无)"))

        # table list
        md.append(u"- 表格清单（OCR 检出 {} 张）:".format(n_tables))
        table_pages = {}
        for i, t in enumerate(tables):
            p = page_of(t)
            q = quality(t)
            pv = preview(t)
            empty = (not pv.strip())
            low_q = (q is not None and q < 0.3)
            note = u""
            if empty:
                note = u" [空内容]"
            if low_q:
                note += u" [低质量 parse_quality={}]".format(q)
            if p is not None:
                table_pages.setdefault(p, []).append(i)
            md.append(u"  - T{} page={} pqs={} {} : {}".format(i, p, u"{}".format(q) if q is not None else u"-", note, pv))

        # block list
        md.append(u"- numeric 块清单（{} 个）:".format(n_blocks))
        block_pages = {}
        for i, b in enumerate(blocks):
            p = page_of(b)
            if p is not None:
                block_pages.setdefault(p, []).append(i)
            md.append(u"  - B{} page={} : {}".format(i, p, preview(b)))

        # exclusion analysis
        excluded = []
        if table_pages and block_pages:
            all_pages = sorted(set(table_pages.keys()) | set(block_pages.keys()))
            for p in all_pages:
                nt = len(table_pages.get(p, []))
                nb = len(block_pages.get(p, []))
                if nt > nb:
                    for i in table_pages[p]:
                        t = tables[i]
                        excluded.append({"page": p, "table_idx": i,
                                         "pqs": quality(t),
                                         "preview": preview(t, 80)})
            reason = u"按页匹配：有表格但无对应数字块的页 → 该页表格视为被排除"
        else:
            # no usable page info on one side: fall back to empty/low-quality heuristic
            for i, t in enumerate(tables):
                pv = preview(t)
                q = quality(t)
                if (not pv.strip()) or (q is not None and q < 0.3):
                    excluded.append({"page": page_of(t), "table_idx": i,
                                     "pqs": q, "preview": pv})
            reason = u"无可用页码字段做精确匹配，以下为疑似被排除候选（空内容/低质量）"

        if excluded:
            md.append(u"- **被排除分析**: " + reason)
            for e in excluded:
                md.append(u"  - page={} T{} pqs={} : {}".format(
                    e["page"], e["table_idx"], u"{}".format(e["pqs"]) if e["pqs"] is not None else u"-", e["preview"]))
        else:
            md.append(u"- **被排除分析**: 未发现（全部表格都有对应数字块）")

        md.append(u"")

        note = u""
        if excluded:
            note = u"{} 张疑似被排除".format(len(excluded))
        else:
            note = u"无异常"
        summary_rows.append(u"| {} | {} | {} | {} | {} |".format(pdf, n_tables, n_blocks, len(excluded), note))

        manifest["samples"].append({"pdf": pdf, "n_tables": n_tables, "n_blocks": n_blocks,
                                    "excluded_count": len(excluded), "excluded": excluded})

    md = md[:8] + summary_rows + [u""] + md[8:]

    with open(os.path.join(rundir, "report.json"), "w") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=True))
    io.open(os.path.join(rundir, "report.md"), "w", encoding="utf-8").write(u"\n".join(md))

    print(u"Report generated: " + os.path.abspath(rundir))
    print(u"Summary:")
    for s in manifest["samples"]:
        print(u"  {}  tables={}  blocks={}  excluded={}".format(
            s["pdf"], s["n_tables"], s["n_blocks"], s["excluded_count"]))


if __name__ == "__main__":
    main()
