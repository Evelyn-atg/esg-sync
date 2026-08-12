"""
LLM Variable Matcher: takes numeric blocks extracted from a PDF and uses an LLM
to match them against the full ESG indicator variable list.

Input:  numeric_extracts/<pdf_name>/numeric_blocks.json
Config: quantitative_variables.json (quantitative variable definitions per indicator)
Output: quantitative_results_Qwen/<pdf_name>/<pdf_name>_quantitative_analysis.json
        (compatible with existing calculator downstream)
"""

import json
import logging
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import Config
from src.utils import setup_logger, ensure_directory_exists, save_json

logger = setup_logger(__name__, "logs/llm_variable_matcher.log")

# Maximum blocks to send per LLM call (control token usage)
_BLOCKS_PER_BATCH = 3
# Maximum content length per block sent to LLM (truncate very long table pages)
_MAX_BLOCK_CONTENT_LENGTH = 6000


class LLMVariableMatcher:
    """Matches numeric blocks to ESG indicator variables using an LLM."""

    def __init__(self):
        self.api_key = Config.QWEN_MAX_API_KEY
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.quantitative_variables_file = Config.QUANTITATIVE_VARIABLES_FILE
        self._variable_list_cache: Optional[List[Dict[str, Any]]] = None

    def _load_variable_list(self) -> List[Dict[str, Any]]:
        """Load and cache the extractable variable list from quantitative_variables.json."""
        if self._variable_list_cache is not None:
            return self._variable_list_cache

        with open(self.quantitative_variables_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        variable_list = []
        for indicator in data.get("indicators", []):
            extractable = (indicator.get("variables") or {}).get("extractable", [])
            if not extractable:
                continue
            variable_list.append({
                "id": indicator.get("id", ""),
                "metric": indicator.get("metric", ""),
                "category": indicator.get("category", ""),
                "variables": extractable,
            })

        self._variable_list_cache = variable_list
        logger.info(f"Loaded {len(variable_list)} quantitative indicators")
        return variable_list

    def _build_variable_reference(self) -> str:
        """Build a concise variable reference string for the LLM prompt."""
        variable_list = self._load_variable_list()
        lines = []
        for item in variable_list:
            var_str = "; ".join(item["variables"])
            lines.append(f"- {item['id']} {item['metric']}: variables=[{var_str}]")
        return "\n".join(lines)

    def _truncate_block_content(self, content: str) -> str:
        """Truncate block content for LLM prompt. Only truncate extremely long blocks."""
        # Generous limit — most blocks under 10K are fine. Only hard-cap at 12K.
        max_len = 12000
        if len(content) <= max_len:
            return content
        # For very long blocks, keep first 8K + last 4K (tables usually have data at start)
        return content[:8000] + "\n...[truncated]...\n" + content[-4000:]

    def _build_batch_prompt(
        self,
        blocks: List[Dict[str, Any]],
        variable_reference: str,
        reporting_year: Optional[int] = None,
    ) -> str:
        """Build the LLM prompt for a batch of numeric blocks."""
        year_instruction = ""
        if reporting_year:
            year_instruction = (
                f"\n报告默认年份: {reporting_year}。"
                f"如果文本中没有明确年份标注，使用 {reporting_year} 作为 year。"
                f"如果文本明确标注了其他年份，以文本标注为准。"
            )

        blocks_text = ""
        for i, block in enumerate(blocks):
            content = self._truncate_block_content(block["content"])
            source_label = "表格页" if block.get("has_table") else "文本段落"
            blocks_text += f"\n[Block {i+1}, Page {block['page']}, {source_label}]:\n{content}\n"

        prompt = f"""你是ESG数据提取助手。以下是从ESG报告中提取的含数值文本块。
请将它们匹配到下方Variable List中对应的指标，并提取具体数值。
{year_instruction}

## Variable List (按指标编号排列):
{variable_reference}

## 文本块:
{blocks_text}

## 要求:
1. 只提取与Variable List中指标明确相关的数值
2. 忽略无关数字（页码、目录、无关财务数据等）
3. 如果一个文本块包含多个variable的数值，分别列出
4. 如果没有找到任何匹配，返回空数组
5. GHG同义词: "直接排放"=Scope 1, "间接排放/能源间接排放"=Scope 2, "其他间接排放"=Scope 3
6. 注意单位换算: 万吨→吨 (×10000), 千吨→吨 (×1000), 兆瓦时→千瓦时 (×1000)
7. Scope判断: 如果报告只提到"CO2 emissions"而没有标注Scope 1/2/3，根据上下文判断:
   - 来自电力消耗(purchased electricity/power usage)的排放 = Scope 2 (indirect emissions)
   - 来自燃料燃烧(fuel/gas/vehicle)的排放 = Scope 1 (direct emissions)
   - 如果报告声明"no direct energy consumption"或"no Scope 1"，则全部CO2排放归为Scope 2
   - 匹配到variable="greenhouse gas emission of Scope 1 (direct emissions) and Scope 2 (indirect emissions)"
8. GHG Scope拆分 (重要): 当文本中出现Scope 1和Scope 2的**单独数值**时，必须分别提取:
   - Scope 1单独数值 → variable="greenhouse gas emission of Scope 1 (direct emissions)", indicator_id="E1.1.1"
   - Scope 2单独数值 → variable="greenhouse gas emission of Scope 2 (indirect emissions)", indicator_id="E1.1.1"
   - Scope 3单独数值 → variable="greenhouse gas emission of Scope 3", indicator_id="E1.1.2"
   - 如果同时有合计值(Scope 1+2 total)，也单独提取为variable="greenhouse gas emission of Scope 1 (direct emissions) and Scope 2 (indirect emissions)", indicator_id="E1.1.1"
   - 即: 同一个表格可能产出3-4条记录(S1, S2, S3, S1+2 total)
9. 多年数据: 如果表格中有多年列(如2022/2023/2024)，每年每个指标都要单独提取一条记录。不要只取最新年份。
10. 年份推断: 如果文本标注"fiscal year ended March 2025"或"2024/2025"，数据的year填写该财年的结束年份(即2025)。
11. 零值提取: 当报告明确声明某指标为零时，也要提取。例如:
    - "no work-related fatalities/injuries" → Number of Fatalities = 0, Number of Lost Time Injuries = 0
    - "no hazardous waste generated" → Hazardous Waste Generated = 0
    - "zero cases of data breach" → Number of Data Breach Incidents = 0
    value填0, confidence=0.9
12. 员工流失率: 当表格列出turnover rate/流失率(百分比)时:
    - 总流失率 → variable="Employee Turnover Rate this year", indicator_id="S1.1.1"
    - 如果只有分性别数据没有总数，取所有分类中最高值或加权平均作为总流失率
    - 分性别/年龄/地区的子项也要提取，variable名加后缀如"Employee Turnover Rate this year (Male)"
13. 能源消耗子项: 当表格拆分不同能源类型时(如电力/天然气/柴油/汽油)，分别提取:
    - 总能耗 → variable="Total Energy Consumption (converted to tce)", indicator_id="E1.3.1"
    - 直接能耗(direct/fuel) → variable="Direct Energy Consumption", indicator_id="E1.3.1"
    - 间接能耗(indirect/electricity) → variable="Indirect Energy Consumption", indicator_id="E1.3.1"
    - 可再生能源 → variable="Renewable Energy Consumption (from self-generation and off-site purchase)", indicator_id="E1.3.3"

## 返回格式 (严格JSON):
[{{"indicator_id": "E1.1.1", "variable": "变量名", "value": 数值, "unit": "单位", "year": 年份或null, "source_page": 页码, "confidence": 0.0-1.0}}]

如果没有任何匹配，返回: []"""

        return prompt

    def _call_llm(self, prompt: str) -> tuple:
        """Call the Qwen text LLM API with retry. Returns (content_str, usage_dict).

        When Config.ENABLE_THINKING is True, adds enable_thinking to the API
        parameters so the model reasons through complex indicator transformations
        (e.g. unit conversions, ratio derivations, multi-step lookups).
        The reasoning_content in the response is logged but not returned;
        only the final answer (content) is used for parsing.
        """
        if not self.api_key:
            logger.error("QWEN_MAX_API_KEY not set")
            return "", {}

        # Build parameters — extend when thinking mode is on
        params = {
            "max_tokens": 8000,   # 大KPI表(30行×多年≈90条记录)输出远超2000，会被截断丢批
            "temperature": 0.1,
        }

        thinking_enabled = getattr(Config, 'ENABLE_THINKING', False)
        if thinking_enabled:
            thinking_budget = getattr(Config, 'THINKING_BUDGET', 8192)
            thinking_max = getattr(Config, 'THINKING_MAX_TOKENS', 16384)
            # Per DashScope: max_tokens includes BOTH reasoning_content and content.
            # thinking_budget caps the reasoning; max_tokens caps the total.
            # Formula: max_tokens = thinking_budget + answer_budget.
            params["max_tokens"] = thinking_max
            params["enable_thinking"] = True
            params["thinking_budget"] = thinking_budget
            logger.info(f"[thinking] enable_thinking=True, thinking_budget={thinking_budget}, max_tokens={thinking_max}, model={Config.QWEN_MAX_MODEL}")

        payload = {
            "model": Config.QWEN_MAX_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        payload.update(params)

        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # When thinking is on, generation takes much longer (CoT reasoning).
                # Increase read timeout from 300s to 600s to accommodate deep thinking.
                read_timeout = 600 if thinking_enabled else 300
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=(10, read_timeout),
                )
                response.raise_for_status()
                result = response.json()

                usage = result.get("usage", {})

                # Extract text from response
                choices = result.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    # When thinking is enabled, the response may contain
                    # reasoning_content (the CoT trace) separate from content.
                    # Log reasoning_content for debugging but only return content.
                    reasoning = message.get("reasoning_content", "")
                    if reasoning and thinking_enabled:
                        reasoning_preview = reasoning[:200] if isinstance(reasoning, str) else str(reasoning)[:200]
                        logger.info(f"[thinking] reasoning preview: {reasoning_preview}...")
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content, usage
                    if isinstance(content, list):
                        text = " ".join(
                            item.get("text", "") for item in content if isinstance(item, dict)
                        )
                        return text, usage
                return "", usage
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)
                    logger.warning(f"LLM API SSL/connection error (attempt {attempt+1}/{max_retries}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM API call failed after {max_retries} attempts: {e}")
                    return "", {}
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                return "", {}

    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the LLM response into structured matches."""
        if not response_text:
            return []

        # Try to extract JSON array from response
        # Look for code block first
        code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a JSON array directly
        bracket_start = response_text.find('[')
        if bracket_start == -1:
            return []

        bracket_count = 0
        for i, char in enumerate(response_text[bracket_start:], bracket_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    try:
                        return json.loads(response_text[bracket_start:i + 1])
                    except json.JSONDecodeError:
                        break

        # 数组未闭合(响应被 max_tokens 截断) → 抢救已生成的完整对象，别整批丢弃
        salvaged = self._salvage_truncated_array(response_text[bracket_start:])
        if salvaged:
            logger.warning(
                f"LLM响应疑似被截断(JSON数组未闭合)，已抢救 {len(salvaged)} 条完整记录。"
                f"若频繁出现，考虑再调大 max_tokens 或缩小批次。"
            )
        else:
            logger.warning("LLM响应无法解析(可能截断且无完整对象可救)，返回空。")
        return salvaged

    def _salvage_truncated_array(self, text: str) -> List[Dict[str, Any]]:
        """从被截断的 JSON 数组文本里逐个抢救完整的 {..} 对象。"""
        objs = []
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        objs.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = -1
        return objs

    def _infer_reporting_year(self, pdf_name: str) -> Optional[int]:
        """Try to infer the reporting year from existing text extraction or reporting period context."""
        from src.image_recognizer import ImageRecognizer
        recognizer = ImageRecognizer()
        text_content = recognizer._load_pdf_text_content(pdf_name)
        if not text_content:
            return None

        context = recognizer.extract_reporting_period_context(pdf_name)
        return context.get("default_year")

    def match_variables_for_pdf(
        self,
        pdf_name: str,
        numeric_blocks: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry: match numeric blocks to variables for a single PDF.
        Returns result in quantitative_analysis.json compatible format.
        """
        output_dir = Config.QUANTITATIVE_RESULT_DIR / pdf_name
        output_file = output_dir / f"{pdf_name}_quantitative_analysis.json"

        if not force and output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f"[{pdf_name}] Skipping — existing quantitative analysis found")
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Load numeric blocks if not provided
        if numeric_blocks is None:
            blocks_file = Config.NUMERIC_EXTRACT_DIR / pdf_name / "numeric_blocks.json"
            if not blocks_file.exists():
                logger.error(f"[{pdf_name}] No numeric blocks file found: {blocks_file}")
                return {}
            with open(blocks_file, 'r', encoding='utf-8') as f:
                numeric_blocks = json.load(f)

        if not numeric_blocks:
            logger.warning(f"[{pdf_name}] No numeric blocks to process")
            return {}

        # Build variable reference (same for all batches)
        variable_reference = self._build_variable_reference()

        # Infer reporting year
        reporting_year = self._infer_reporting_year(pdf_name)
        logger.info(f"[{pdf_name}] Inferred reporting year: {reporting_year}")

        # Build smart batches: large blocks get their own batch, small blocks group together
        batches = []
        current_batch = []
        current_batch_chars = 0
        _MAX_BATCH_CHARS = 15000  # Max total content chars per batch

        _BIG_TABLE_CHARS = 5000  # 密集KPI表：多年多行→产出大量记录，单独成批避免挤在一批被截断

        for block in numeric_blocks:
            block_len = len(block.get("content", ""))
            is_big_table = block.get("has_table") and block_len > _BIG_TABLE_CHARS
            # 单块超批上限，或本身是大表格页 → 独占一批
            if block_len > _MAX_BATCH_CHARS or is_big_table:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_chars = 0
                batches.append([block])
            # If adding this block would exceed limit, start a new batch
            elif current_batch_chars + block_len > _MAX_BATCH_CHARS or len(current_batch) >= _BLOCKS_PER_BATCH:
                batches.append(current_batch)
                current_batch = [block]
                current_batch_chars = block_len
            else:
                current_batch.append(block)
                current_batch_chars += block_len
        if current_batch:
            batches.append(current_batch)

        all_matches: List[Dict[str, Any]] = []
        total_batches = len(batches)
        total_input_tokens = 0
        total_output_tokens = 0

        max_workers = min(total_batches, Config.LLM_MATCHING_MAX_WORKERS)
        logger.info(f"[{pdf_name}] Processing {total_batches} batches with {max_workers} workers")

        def _process_batch(batch_idx: int):
            batch = batches[batch_idx]
            batch_num = batch_idx + 1
            prompt = self._build_batch_prompt(batch, variable_reference, reporting_year)
            response, usage = self._call_llm(prompt)
            matches = self._parse_llm_response(response)
            return batch_num, matches, usage

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_batch, idx): idx
                for idx in range(total_batches)
            }
            for future in as_completed(futures):
                batch_num, matches, usage = future.result()
                total_input_tokens += usage.get("input_tokens", 0)
                total_output_tokens += usage.get("output_tokens", 0)
                if matches:
                    all_matches.extend(matches)
                    logger.info(f"[{pdf_name}] Batch {batch_num}/{total_batches}: {len(matches)} matches found")
                else:
                    logger.info(f"[{pdf_name}] Batch {batch_num}/{total_batches}: no matches")

        # Print cost summary
        # qwen3.8-max: input ¥12.00/1M tokens, output ¥36.00/1M tokens
        # qwen3.7-max: input ¥2.00/1M tokens, output ¥8.00/1M tokens
        # qwen-max:    input ¥2.00/1M tokens, output ¥6.00/1M tokens
        model_name = Config.QWEN_MAX_MODEL
        if "3.8" in model_name:
            input_price, output_price = 12.00, 36.00
        elif "3.7" in model_name:
            input_price, output_price = 2.00, 8.00
        else:
            input_price, output_price = 2.00, 6.00
        input_cost = total_input_tokens / 1_000_000 * input_price
        output_cost = total_output_tokens / 1_000_000 * output_price
        total_cost = input_cost + output_cost
        logger.info(
            f"[{pdf_name}] Token usage: input={total_input_tokens}, output={total_output_tokens}, "
            f"cost=¥{total_cost:.4f} (in=¥{input_cost:.4f} + out=¥{output_cost:.4f}) [{Config.QWEN_MAX_MODEL}]"
        )

        # Convert matches to quantitative_analysis.json format
        result = self._format_output(pdf_name, all_matches, reporting_year)

        # Save
        ensure_directory_exists(output_dir)
        save_json(result, output_file)
        logger.info(f"[{pdf_name}] Saved {len(all_matches)} total matches to {output_file}")

        return result

    def _format_output(
        self,
        pdf_name: str,
        matches: List[Dict[str, Any]],
        reporting_year: Optional[int],
    ) -> Dict[str, Any]:
        """Format LLM matches into the existing quantitative_analysis.json structure."""
        # Group matches by indicator
        all_variables_collected: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for match in matches:
            indicator_id = match.get("indicator_id", "unknown")
            variable = match.get("variable", "unknown")
            value = match.get("value")
            unit = match.get("unit", "")
            year = match.get("year")
            page = match.get("source_page")
            confidence = match.get("confidence", 0.5)

            # Find the indicator metric name from the variable list
            indicator_metric = indicator_id
            for item in self._load_variable_list():
                if item["id"] == indicator_id:
                    indicator_metric = item["metric"]
                    break

            if indicator_metric not in all_variables_collected:
                all_variables_collected[indicator_metric] = {}

            if variable not in all_variables_collected[indicator_metric]:
                all_variables_collected[indicator_metric][variable] = []

            all_variables_collected[indicator_metric][variable].append({
                "indicator": variable,
                "value": value,
                "unit": unit,
                "year": year,
                "raw_year": year,
                "source_page": page,
                "confidence": confidence,
                "source_type": "llm_matched",
                "indicator_id": indicator_id,
            })

        return {
            "pdf_name": pdf_name,
            "all_variables_collected": all_variables_collected,
            "calculation_formulas": {},
            "reporting_period_context": {
                "default_year": reporting_year,
                "resolution_method": "inferred",
            },
            "enhancement_metadata": {
                "method": "llm_variable_matching",
                "total_matches": len(matches),
                "processing_completed": True,
            },
        }

    def process_specific_pdfs(
        self,
        pdf_paths: List[Path],
        force: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Process LLM variable matching for a list of PDFs."""
        results = {}
        for pdf_path in pdf_paths:
            pdf_name = pdf_path.stem
            try:
                result = self.match_variables_for_pdf(pdf_name, force=force)
                results[pdf_name] = result
            except Exception as e:
                logger.error(f"Failed LLM variable matching for {pdf_name}: {e}", exc_info=True)
                results[pdf_name] = {}
        return results
