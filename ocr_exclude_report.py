#!/usr/bin/env python
# -*- coding: utf-8 -*-
# v3: run against a STABLE snapshot dir (--from-dir), classify OCR tables by
# content (real table vs chart/image vs other), dump block source values to
# reveal linkage, page-level real-table vs has_table-block coverage.
# py2.7-safe: no f-strings, u"" literals, json via ensure_ascii=True.
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


def content_of(t):
    if isinstance(t, dict):
        for k in ("content_html", "html", "content", "markdown", "text"):
            if t.get(k):
                return u"{}".format(t[k])
    return u""


def classify(t):
    c = content_of(t).lower()
    if u"<table" in c:
        return u"table"
    if u"<img" in c:
        return u"chart_img"
    if not c.strip():
        return u"empty"
    return u"other"


def preview(x, n=80):
    if isinstance(x, dict):
        for k in ("content_html", "html", "content", "text", "markdown"):
            if x.get(k):
                return u"{}".format(x[k]).replace(u"\n", u" ")[:n]
        return json.dumps(x, ensure_ascii=True)[:n]
    return u"{}".format(x)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--from-dir", default=None,
                    help="run against an existing snapshot dir (must contain ocr/ and numeric/ subdirs)")
    ap.add_argument("--ocr-root", default="quantitative_results_ocr/chandra_ocr_2")
    ap.add_argument("--num-root", default="numeric_extracts")
    ap.add_argument("--out", default="spotcheck_report")
    args = ap.parse_args()

    if args.from_dir:
        ocr_root = os.path.join(args.from_dir, "ocr")
        num_root = os.path.join(args.from_dir, "numeric")
        snap = True
    else:
        ocr_root, num_root, snap = args.ocr_root, args.num_root, False

    if not os.path.isdir(ocr_root):
        sys.exit(u"OCR root not found: {}\n(run this from inside ~/esg-pipeline)".format(ocr_root))

    if snap:
        pdfs = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(ocr_root, "*.json")))
        chosen = pdfs[:args.n]
    else:
        pdf_dirs = sorted(p for p in glob.glob(os.path.join(ocr_root, "*")) if os.path.isdir(p))
        random.seed(args.seed)
        n = min(args.n, len(pdf_dirs))
        chosen = [os.path.basename(p) for p in random.sample(pdf_dirs, n)]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rundir = os.path.join(args.out, "report_" + ts)
    if not os.path.isdir(rundir):
        os.makedirs(rundir)

    md = [u"# OCR 抽样核对报告 v3（内容分类 + 页码覆盖，基于快照）", u"",
          u"- 时间: " + ts, u"- 来源: " + (u"快照 " + args.from_dir if snap else u"实时目录"),
          u"- 抽样数: {}".format(len(chosen)), u"- ocr_root: `{}`".format(ocr_root),
          u"- num_root: `{}`".format(num_root), u""]

    summary_rows = [u"| PDF | 表区域 | 真表格 | 图表/图片 | 其他/空 | 块数 | has_table块 | 块source值 | 页码级疑似缺失页 |",
                    u"|---|---|---|---|---|---|---|---|---|"]
    manifest = {"generated_at": ts, "n_total": len(chosen), "n_sampled": len(chosen), "samples": []}

    for pdf in chosen:
        if snap:
            ocr_file = os.path.join(ocr_root, pdf + ".json")
            num_file = os.path.join(num_root, pdf + ".json")
        else:
            ocr_file = os.path.join(ocr_root, pdf, pdf + "_ocr_output.json")
            num_file = os.path.join(num_root, pdf, "numeric_blocks.json")

        ocr = load(ocr_file)
        tables = find_tables(ocr)
        num = load(num_file)
        blocks = pick_blocks(num)

        md.append(u"## " + pdf)
        if "__error__" in ocr or "__error__" in num:
            md.append(u"- 读取错误: ocr={} num={}".format(ocr.get("__error__"), num.get("__error__")))
            md.append(u"")
            continue

        # classify OCR tables
        cat = {u"table": 0, u"chart_img": 0, u"other": 0, u"empty": 0}
        real_pages = {}
        for i, t in enumerate(tables):
            c = classify(t)
            cat[c] += 1
            if c == u"table":
                p = page_of(t)
                if p is not None:
                    real_pages.setdefault(p, []).append((i, t))

        # blocks: has_table flag + unique source values
        n_blocks = len(blocks)
        n_ht = sum(1 for b in blocks if isinstance(b, dict) and b.get("has_table"))
        sources = []
        ht_pages = {}
        for i, b in enumerate(blocks):
            if isinstance(b, dict):
                s = b.get("source")
                if s is not None:
                    sv = u"{}".format(s)
                    if sv not in sources:
                        sources.append(sv)
                if b.get("has_table"):
                    p = page_of(b)
                    if p is not None:
                        ht_pages.setdefault(p, []).append(i)

        # page-level: pages with real tables exceeding has_table blocks
        suspect_pages = []
        for p in sorted(set(real_pages.keys()) | set(ht_pages.keys())):
            nt = len(real_pages.get(p, []))
            nh = len(ht_pages.get(p, []))
            if nt > nh:
                suspect_pages.append({"page": p, "real_tables": nt, "has_table_blocks": nh,
                                      "tables": [preview(t) for _, t in real_pages[p]]})

        md.append(u"- 表区域 {} 个：真表格 {} / 图表图片 {} / 其他 {} / 空 {}".format(
            len(tables), cat[u"table"], cat[u"chart_img"], cat[u"other"], cat[u"empty"]))
        md.append(u"- numeric 块 {} 个，has_table={}".format(n_blocks, n_ht))
        md.append(u"- 块 source 唯一值 ({}): {}".format(len(sources), u" | ".join(sources) if sources else u"(空)"))
        if suspect_pages:
            md.append(u"- **页码级疑似缺失**（该页真表格数 > has_table 块数）:")
            for sp in suspect_pages:
                md.append(u"  - page={}: 真表格 {} vs 块 {} → 表: {}".format(
                    sp["page"], sp["real_tables"], sp["has_table_blocks"], u" || ".join(sp["tables"])))
        else:
            md.append(u"- 页码级无缺失（真表格均有对应 has_table 块）")
        md.append(u"")

        summary_rows.append(u"| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            pdf, len(tables), cat[u"table"], cat[u"chart_img"], cat[u"other"] + cat[u"empty"],
            n_blocks, n_ht, u",".join(sources)[:40] or u"-", len(suspect_pages)))
        manifest["samples"].append({"pdf": pdf, "n_tables": len(tables), "categories": cat,
                                    "n_blocks": n_blocks, "n_has_table": n_ht,
                                    "sources": sources, "suspect_pages": suspect_pages})

    md = md[:8] + summary_rows + [u""] + md[8:]

    with open(os.path.join(rundir, "report.json"), "w") as f:
        f.write(json.dumps(manifest, indent=2, ensure_ascii=True))
    io.open(os.path.join(rundir, "report.md"), "w", encoding="utf-8").write(u"\n".join(md))

    print(u"== Summary ==")
    print(u"  {:<16} {:>4} {:>4} {:>4} {:>4} {:>5} {:>4} {:>6}".format(
        u"PDF", u"表", u"真表", u"图表", u"其他", u"块", u"ht", u"疑缺页"))
    for s in manifest["samples"]:
        print(u"  {:<16} {:>4} {:>4} {:>4} {:>4} {:>5} {:>4} {:>6}".format(
            s["pdf"], s["n_tables"], s["categories"][u"table"], s["categories"][u"chart_img"],
            s["categories"][u"other"] + s["categories"][u"empty"],
            s["n_blocks"], s["n_has_table"], len(s["suspect_pages"])))
    print(u"== Unique block source values ==")
    for s in manifest["samples"]:
        print(u"  [{}] {}".format(s["pdf"], u" | ".join(s["sources"]) if s["sources"] else u"(no source field)"))
    print(u"== 页码级疑似缺失（真表格数 > has_table 块数）==")
    for s in manifest["samples"]:
        for sp in s["suspect_pages"]:
            print(u"  [{}] page={} 真表格 {} vs 块 {} : {}".format(
                s["pdf"], sp["page"], sp["real_tables"], sp["has_table_blocks"],
                u" || ".join(sp["tables"])[:120]))
    print(u"Report files: " + os.path.abspath(rundir))


if __name__ == "__main__":
    main()
