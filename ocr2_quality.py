#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""chandra_ocr_2 OCR 结果质量统计（py2.7 + py3 兼容，集群 login1 可用）。

用法:
    python ocr2_quality.py [chandra_ocr_2目录] [--csv 输出.csv]

每报告维度: 文件大小 / tables数 / OCR页数 / 表格html字符 / parse_quality /
            runtime(OCR耗时) / section_hierarchy覆盖率 / markdown与text缺失率
"""
from __future__ import print_function, division
import os
import sys
import io
import json
import glob

ROOT = sys.argv[1] if len(sys.argv) > 1 else "quantitative_results_ocr/chandra_ocr_2"
CSV = None
if "--csv" in sys.argv:
    CSV = sys.argv[sys.argv.index("--csv") + 1]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    pos = (n - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _fmt(v):
    if v >= 1000:
        return "%.0f" % v
    return "%.0f" % v if v == int(v) else "%.1f" % v


def summarize(name, vals):
    if not vals:
        print("  %-20s (无数据)" % name)
        return
    s = sorted(vals)
    mean = sum(s) / len(s)
    line = "  %-20s n=%-6d mean=%-8s | min=%s p25=%s p50=%s p75=%s p95=%s max=%s" % (
        name, len(s), _fmt(mean), _fmt(percentile(s, 0)), _fmt(percentile(s, 25)),
        _fmt(percentile(s, 50)), _fmt(percentile(s, 75)), _fmt(percentile(s, 95)),
        _fmt(percentile(s, 100)))
    print(line)
    return s


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*_ocr_output.json")))
    total = len(files)
    print("扫描目录: %s" % ROOT)
    print("找到 *_ocr_output.json: %d 个\n" % total)

    n_tab_a, size_a, page_a, html_a, q_a, rt_a = [], [], [], [], [], []
    corrupt, empty_tab, empty_pg = [], [], []
    sec_have, sec_total, no_md, no_text = 0, 0, 0, 0
    rows = []
    per_tab = []

    for jf in files:
        pid = os.path.basename(os.path.dirname(jf))
        size = os.path.getsize(jf)
        try:
            with io.open(jf, "r", encoding="utf-8") as f:
                p = json.load(f)
        except Exception:
            corrupt.append(pid)
            rows.append((pid, size, -1, 0, 0, None, None))
            continue

        tables = p.get("tables") or []
        if isinstance(tables, dict):
            tables = [t for t in tables.values() if isinstance(t, dict)] or []
        pages = p.get("pages") or []
        if isinstance(pages, dict):
            n_pg = len(pages)
            pg_list = []
        else:
            n_pg = len(pages)
            pg_list = pages

        n_tab = len(tables)
        hlen = 0
        for t in tables:
            if not isinstance(t, dict):
                continue
            hlen += len(t.get("content_html") or t.get("html") or "")
            sec_total += 1
            if t.get("section_hierarchy"):
                sec_have += 1
            if not (t.get("markdown") or ""):
                no_md += 1
            if not (t.get("text") or ""):
                no_text += 1

        qvals = [x.get("parse_quality_score") for x in pg_list
                 if isinstance(x, dict) and x.get("parse_quality_score") is not None]
        rtvals = [x.get("runtime") for x in pg_list
                  if isinstance(x, dict) and x.get("runtime") is not None]

        n_tab_a.append(n_tab)
        size_a.append(size)
        page_a.append(n_pg)
        html_a.append(hlen)
        if qvals:
            q_a.append(sum(qvals) / len(qvals))
        if rtvals:
            rt_a.extend(rtvals)
        if n_tab == 0:
            empty_tab.append(pid)
        if n_pg == 0:
            empty_pg.append(pid)
        per_tab.append((pid, n_tab, size))
        rows.append((pid, size, n_tab, n_pg, hlen,
                     (sum(qvals) / len(qvals)) if qvals else None))

    print("=" * 80)
    print("汇总: 共 %d 份 | 损坏json %d | 空tables %d | 空pages %d" %
          (total, len(corrupt), len(empty_tab), len(empty_pg)))
    print("=" * 80)

    print("\n【分布】(p25/p50/p75 = 四分位)")
    summarize("tables数", n_tab_a)
    summarize("OCR页数", page_a)
    summarize("文件大小KB", [s / 1024.0 for s in size_a])
    summarize("表格html字符", html_a)

    # tables 直方图
    s = sorted(n_tab_a)
    print("    tables数直方图:")
    for a, b, lb in [(0, 0, "0"), (1, 5, "1-5"), (6, 10, "6-10"), (11, 20, "11-20"),
                     (21, 50, "21-50"), (51, 100, "51-100"), (101, 10 ** 9, "100+")]:
        cnt = sum(1 for v in s if a <= v <= b)
        print("     %6s: %6d  %s" % (lb, cnt, "#" * min(cnt, 60)))

    if q_a:
        summarize("parse_quality", q_a)
    if rt_a:
        rt_s = sorted(rt_a)
        print("  %-20s n=%-6d mean=%.1fs | 单页OCR耗时分布: min=%.1fs p50=%.1fs p95=%.1fs max=%.1fs" %
              ("OCR耗时runtime(s)", len(rt_s), sum(rt_s) / len(rt_s),
               rt_s[0], percentile(rt_s, 50), percentile(rt_s, 95), rt_s[-1]))
        est = sum(rt_s) / len(rt_s) * len(page_a)
        print("    估算全量OCR总耗时(单卡): %.1f 小时（均值外推，未计排队）" % (est / 3600.0))

    if sec_total:
        print("  %-20s %d/%d = %.1f%%" % ("section_hierarchy覆盖",
              sec_have, sec_total, 100.0 * sec_have / sec_total))
        print("  %-20s %d/%d = %.1f%%" % ("markdown缺失", no_md, sec_total,
              100.0 * no_md / sec_total))
        print("  %-20s %d/%d = %.1f%%" % ("text缺失", no_text, sec_total,
              100.0 * no_text / sec_total))

    if empty_tab:
        print("\n【空 tables 报告】(%d 个, 前 30):" % len(empty_tab))
        for pid in empty_tab[:30]:
            print("   %s" % pid)
    if corrupt:
        print("\n【损坏 json】(%d 个):" % len(corrupt))
        for pid in corrupt[:30]:
            print("   %s" % pid)

    print("\n【TOP 10 表格最多】")
    for pid, nt, sz in sorted(per_tab, key=lambda x: -x[1])[:10]:
        print("   %-16s tables=%-5d %6.1f KB" % (pid, nt, sz / 1024.0))
    print("\n【TOP 10 文件最大】")
    for pid, nt, sz in sorted(per_tab, key=lambda x: -x[2])[:10]:
        print("   %-16s %6.1f KB  tables=%d" % (pid, sz / 1024.0, nt))

    if CSV:
        with io.open(CSV, "w", encoding="utf-8") as f:
            f.write(u"pid,size_bytes,n_tables,n_pages,html_chars,parse_quality\n")
            for pid, size, nt, npg, hlen, q in rows:
                f.write(u"%s,%d,%d,%d,%d,%s\n" %
                        (pid, size, nt, npg, hlen,
                         u"%.3f" % q if q is not None else u""))
        print(u"\nCSV 已写: %s (%d 行)" % (CSV, len(rows)))


if __name__ == "__main__":
    main()
