# ESG GT 三维验证（gt_verify）

## 目的
对 100 家公司的 Excel 抽取结果做逐行 Ground Truth 验证，衡量抽取质量与 GT 的一致性。

## 方法（行级三维度）
| 维度 | 字段 | 含义 |
|---|---|---|
| 语义匹配 | matched_semantic | 抽取指标与 GT 指标语义是否对应（yes/no） |
| 单位一致 | unit_consistent | 抽取单位与 GT 单位是否一致（yes/no） |
| 数值匹配 | value_matches | 抽取数值与 GT 数值是否一致（yes/no，含 partial） |

三维全 yes = 该行通过。输出：`gt_verify_out/verify_report.csv`。

## 全量结果（2026-08-06 定稿，100 家 Excel）
- 总行数：16,175
- 三维全 yes：13,839（**85.6%**）
  - 语义匹配 yes：15,175（93.8%）
  - 单位一致 yes：14,985（92.6%）
  - 数值匹配 yes：14,038（86.8%，partial 3 行）
- 排除 00388（港交所）后：15,398 行，三维全 yes **87.8%**

## 00388 港交所例外
- 777 行中三维全 yes 仅 41.8%（语义不匹配为主）
- 原因：抽取时页码定位/指标匹配存在系统性偏差，**非公司识别问题**
- 该公司报告指标多（777 行）、页面结构复杂，拉低整体数字

## 相关
- 验证脚本：gt_verify.py / excel_verify.py
- 数据源：Ground_truth/companies/*.xlsx（100 个文件，代码命名）
