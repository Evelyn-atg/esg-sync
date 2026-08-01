#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os, sys, io, json, random, glob, shutil, datetime, argparse


def pick_blocks(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("blocks", "numeric_blocks", "results", "items"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return []


def find_tables(d):
    if not isinstance(d, dict):
        return []
    for k in ("tables", "table_blocks", "ocr_tables"):
        if isinstance(d.get(k), list):
            return d[k]
    if isinstance(d.get("data"), dict) and isinstance(d["data"].get("tables"), list):
        return d["data"]["tables"]
    return []


def summarize_ocr(path):
    try:
        d = json.load(io.open(path, "r", encoding="utf-8"))
    except Exception as e:
        return {"error": u"{}: {}".format(type(e).__name__, e)}
    tables = find_tables(d)
    n_tables = len(tables)
    has_bbox = any(isinstance(t, dict) and "bbox" in t for t in tables)
    pqs = [t.get("parse_quality_score") for t in tables[:3]
           if isinstance(t, dict) and "parse_quality_score" in t]
    preview = u""
    if tables and isinstance(tables[0], dict):
        preview = tables[0].get("html") or tables[0].get("text") or tables[0].get("markdown") or u""
    return {"pages": d.get("num_pages") if isinstance(d, dict) else None,
            "n_tables": n_tables, "has_bbox": has_bbox,
            "parse_quality_scores": pqs, "text_preview": (preview or u"")[:300]}


def summarize_numeric(path):
    try:
        d = json.load(io.open(path, "r", encoding="utf-8"))
    except Exception as e:
        return {"error": u"{}: {}".format(type(e).__name__, e)}
    blocks = pick_blocks(d)
    n = len(blocks)
    has_bbox = any(isinstance(b, dict) and "bbox" in b for b in blocks)
    preview = json.dumps(blocks[0], ensure_ascii=False)[:300] if blocks else u""
    return {"n_blocks": n, "has_bbox": has_bbox, "block_preview": preview}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--ocr-root", default="quantitative_results_ocr/chandra_ocr_2")
    ap.add_argument("--num-root", default="numeric_extracts")
    ap.add_argument("--out", default="spotcheck")
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
    rundir = os.path.join(args.out, "spotcheck_" + ts)
    ocr_dir = os.path.join(rundir, "ocr")
    num_dir = os.path.join(rundir, "numeric")
    if not os.path.isdir(ocr_dir):
        os.makedirs(ocr_dir)
    if not os.path.isdir(num_dir):
        os.makedirs(num_dir)

    manifest = {"generated_at": ts, "seed": args.seed,
                "ocr_root": os.path.abspath(args.ocr_root),
                "num_root": os.path.abspath(args.num_root),
                "n_total": len(pdf_dirs), "n_sampled": n, "samples": []}

    md = [u"# OCR Spot-check - " + ts, u"",
          u"- seed: `{}`".format(args.seed),
          u"- total PDF dirs: {}".format(len(pdf_dirs)),
          u"- sampled: {}".format(n),
          u"- ocr_root: `{}`".format(args.ocr_root),
          u"- num_root: `{}`".format(args.num_root), u""]

    for d in chosen:
        pdf = os.path.basename(d)
        ocr_file = os.path.join(d, pdf + "_ocr_output.json")
        num_file = os.path.join(args.num_root, pdf, "numeric_blocks.json")
        ocr = summarize_ocr(ocr_file)
        if os.path.exists(num_file):
            num = summarize_numeric(num_file)
            shutil.copy(num_file, os.path.join(num_dir, pdf + ".json"))
        else:
            num = {"error": u"numeric_blocks.json NOT FOUND"}
        if os.path.exists(ocr_file):
            shutil.copy(ocr_file, os.path.join(ocr_dir, pdf + ".json"))

        flags = []
        if ("error" not in ocr and ocr.get("n_tables", 0) > 0
                and "error" not in num and num.get("n_blocks", 0) == 0):
            flags.append("OCR has tables but numeric_blocks empty (all filtered out?)")
        if "error" in num:
            flags.append("numeric result missing")

        manifest["samples"].append({"pdf": pdf, "ocr": ocr, "numeric": num, "flags": flags})

        md.append(u"## " + pdf)
        md.append(u"- OCR: tables={}, has_bbox={}, pages={}, pqs={}".format(
            ocr.get("n_tables"), ocr.get("has_bbox"), ocr.get("pages"), ocr.get("parse_quality_scores")))
        md.append(u"- Numeric: blocks={}, has_bbox={}".format(num.get("n_blocks"), num.get("has_bbox")))
        if flags:
            md.append(u"- **FLAGS**: {}".format(flags))
        if ocr.get("text_preview"):
            md.append(u"- OCR preview: {}".format(ocr["text_preview"]))
        if num.get("block_preview"):
            md.append(u"- Numeric preview: {}".format(num["block_preview"]))
        md.append(u"")

    json.dump(manifest, io.open(os.path.join(rundir, "manifest.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    io.open(os.path.join(rundir, "manifest.md"), "w", encoding="utf-8").write(u"\n".join(md))

    print(u"Spot-check complete. {}/{} sampled (seed={}).".format(n, len(pdf_dirs), args.seed))
    print(u"Trace saved under: " + os.path.abspath(rundir))
    print(u"Chosen PDFs:")
    for s in manifest["samples"]:
        print(u"  {}  ocr_tables={}  num_blocks={}  flags={}".format(
            s["pdf"], s["ocr"].get("n_tables"), s["numeric"].get("n_blocks"), s["flags"]))


if __name__ == "__main__":
    main()
