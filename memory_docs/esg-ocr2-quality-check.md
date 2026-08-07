# OCR2 质量检查 Pipeline（chandra_ocr_2）

## 检查目标
在进入 LLM 匹配前，验证 Chandra OCR 结果完整性 + 质量。
对象：`quantitative_results_ocr/chandra_ocr_2/{pid}/{pid}_ocr_output.json`（每报告一个目录）。

## 检查项（一）数字核对：避免 ls 计数陷阱
- `ls -1 | wc -l` 会混入非目录文件（清单 txt、提交脚本、slurm .out）→ 用 `ls -1d */ | wc -l` 数纯目录
- 有目录但无 ocr_output.json = 真缺口；目录里只有 `_tmp_table_crops` 无 json = OCR 中途断
- 差集命令：
```bash
cd <chandra_ocr_2目录>
find . -maxdepth 2 -name "*_ocr_output.json" -exec dirname {} \; | sed 's|^\./||' | sort -u > /tmp/has.txt
ls -1d */ | sed 's|/$||' | sort > /tmp/all.txt
comm -23 /tmp/all.txt /tmp/has.txt     # 真缺口
```

## 检查项（二）ocr2_quality.py 全量统计
- 脚本：esg-sync 仓库根目录 `ocr2_quality.py`（py2.7 + py3 双兼容）
- 用法：`python ocr2_quality.py <chandra_ocr_2目录> --csv /tmp/ocr2_quality.csv`
- 维度：
  - tables 数（quantile + 直方图）
  - OCR 页数 / 文件大小 / 表格 html 字符量
  - parse_quality / OCR 单页 runtime（可外推全量耗时）
  - section_hierarchy 覆盖 / markdown 缺失 / text 缺失
  - 空 tables 清单、损坏 json、TOP 10 榜
- 基线（2026-08-07，9389 份）：0 损坏；tables p25=9/p50=16/p75=24/max=159；文件 p50=76KB/max=1.3MB；空表 179 份
- 字段缺失判定（不影响下游）：
  - markdown/text 100% 缺失 → 无害：`numeric_extractor.py` L267 "Prefer HTML, fall back to markdown"，content_html 全量存在
  - section_hierarchy 0% → 当前 `llm_variable_matcher.py` 不消费；仅后续章节归类需要时再补

## 检查项（三）空表兜底对照（关键判定）
- 空 tables + 已有 `numeric_extracts/{pid}/numeric_blocks.json` = 纯文本报告（正常，走 PyMuPDF 文本层）
- 空 tables + 无 blocks = 真失败（suspicious），需补跑/人工看 PDF
- 基线：179 空表全部有 blocks，suspicious = 0 → OCR 层质量闭环

## 已知坑
- login1 系统 python 2.7.5：heredoc 传中文脚本需第一行 `# -*- coding: utf-8 -*-`（PEP 263）；CSV 写入必须 unicode（`u"..."`），否则 `TypeError: must be unicode, not str`
- `--force` 不会重跑已有页（OCR 逐页缓存，`pages_to_ocr = [p for p in page_numbers if p not in cached]`）
- 提交清单格式必须是 `pid.pdf`（`main.py load_pdf_names_from_list_file` 直接拿行内容当 pdf_name）
- sed 改脚本内绝对路径时，替换必须覆盖完整路径段，否则残留目录段导致 "PDF list file not found"
- 补跑用 `run_gpu*.sh`（-p A800，`--step numeric_extraction`）：该 step 是原子步骤 = 表格页 OCR + 提取 blocks，两步一起完成
