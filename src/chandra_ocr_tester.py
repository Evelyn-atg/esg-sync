import json
import re
import shutil
import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

from PIL import Image

from src.config import Config
from src.image_recognizer import ImageRecognizer
from src.utils import ensure_directory_exists, save_json, setup_logger

logger = setup_logger(__name__, "logs/chandra_ocr_tester.log")


class ChandraOCRTester:
    """
    Standalone OCR tester.

    Final retained outputs per PDF:
    1. OCR JSON output
    2. Qwen-normalized quantitative result in the same structure as the original VLM path

    Temporary page images and table crops are deleted after use.
    """

    RESULT_FORMAT_VERSION = "datalab_convert_api_quantitative_v3"
    # Keep the best value per scope PER YEAR, up to this many distinct years. A flat
    # top-N cap dropped the oldest column of multi-year tables (2902036: 2022/2023/2024
    # → 2022 trimmed); per-year dedup keeps all years a report discloses.
    MAX_YEARS_PER_VARIABLE = 6

    def __init__(self, convert_mode: str | None = None):
        self.image_recognizer = ImageRecognizer()
        self.convert_mode = convert_mode or Config.DATALAB_CONVERT_MODE
        self.ocr_backend = Config.OCR_BACKEND
        self.reuse_cached_ocr = Config.REUSE_CACHED_OCR
        # Set while re-running downstream from cached OCR: recognition has no table
        # crop image and runs text-only off the cached table HTML.
        self.text_only_recognition = False
        self._client = None
        self._chandra_manager = None

    def _get_pdf_output_dir(self, pdf_name: str) -> Path:
        return Config.CHANDRA_OCR_RESULT_DIR / pdf_name

    def _get_pdf_ocr_output_file(self, pdf_name: str) -> Path:
        return self._get_pdf_output_dir(pdf_name) / f"{pdf_name}_ocr_output.json"

    def _get_pdf_quantitative_result_file(self, pdf_name: str) -> Path:
        return self._get_pdf_output_dir(pdf_name) / f"{pdf_name}_quantitative_analysis.json"

    def _get_temp_table_crops_dir(self, pdf_name: str) -> Path:
        return self._get_pdf_output_dir(pdf_name) / "_tmp_table_crops"

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        if not Config.DATALAB_API_KEY:
            raise ValueError(
                "DATALAB_API_KEY is not set. Add it to your environment or .env file before running OCR."
            )

        try:
            from datalab_sdk import DatalabClient
        except ImportError as exc:
            raise ImportError(
                "Datalab SDK is missing. Install it with: pip install -r requirements.txt"
            ) from exc

        self._client = DatalabClient(api_key=Config.DATALAB_API_KEY)
        return self._client

    def _load_keywords_config(self, keywords_config_path: Path) -> Dict[str, Any]:
        with open(keywords_config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_variable_definitions(self, variable_definitions_path: Path) -> Dict[str, Any]:
        with open(variable_definitions_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        definitions = payload.get("variables", payload)
        if not isinstance(definitions, dict):
            raise ValueError(f"Invalid variable definitions structure in {variable_definitions_path}")
        return definitions

    def _extract_quantitative_indicator_maps(
        self,
        keywords_config: Dict[str, Any],
    ) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        sub_indicator_variables_map: Dict[str, List[str]] = {}
        sub_indicator_formulas_map: Dict[str, List[str]] = {}

        for indicator_data in keywords_config.get("indicators", []):
            if "sub_indicators" in indicator_data:
                for sub_indicator in indicator_data["sub_indicators"]:
                    sub_indicator_name = sub_indicator["name"]
                    variables = sub_indicator.get("variables", [])
                    formulas = sub_indicator.get("calculation_formulas", [])
                    if variables or formulas:
                        sub_indicator_variables_map[sub_indicator_name] = variables
                        sub_indicator_formulas_map[sub_indicator_name] = formulas
            else:
                indicator_name = indicator_data["name"]
                variables = indicator_data.get("variables", [])
                formulas = indicator_data.get("calculation_formulas", [])
                if variables or formulas:
                    sub_indicator_variables_map[indicator_name] = variables
                    sub_indicator_formulas_map[indicator_name] = formulas

        return sub_indicator_variables_map, sub_indicator_formulas_map

    def _build_variable_definition_page_plan(
        self,
        text_content: str,
        indicator_variables_map: Dict[str, List[str]],
        variable_definitions: Dict[str, Any],
    ) -> Dict[str, Any]:
        page_text_map = self.image_recognizer._parse_page_text_map(text_content)
        candidate_records: List[Dict[str, Any]] = []
        selected_pages: Dict[int, Dict[str, Any]] = {}

        variable_to_indicators: Dict[str, List[str]] = {}
        for indicator_name, variable_names in indicator_variables_map.items():
            for variable_name in variable_names:
                variable_to_indicators.setdefault(variable_name, []).append(indicator_name)

        for variable_name, definition in variable_definitions.items():
            if variable_name not in variable_to_indicators:
                continue

            if not isinstance(definition, dict):
                continue

            primary_keywords = definition.get("page_candidate_keywords", [])
            if not primary_keywords:
                continue

            bonus_keywords = definition.get("page_bonus_keywords", [])
            forbidden_keywords = definition.get("page_forbidden_keywords", [])
            min_bonus_hits = int(definition.get("page_min_bonus_hits", 1) or 0)
            max_candidates = int(definition.get("page_max_candidates", 2) or 2)
            indicator_names = variable_to_indicators.get(variable_name, [])

            scored_pages: List[Dict[str, Any]] = []
            for page_num, page_text in page_text_map.items():
                normalized_page = self._sanitize_text(page_text)
                if not normalized_page:
                    continue

                if self._contains_any_phrase(normalized_page, forbidden_keywords):
                    continue

                primary_hits = self._count_matched_phrases(normalized_page, primary_keywords)
                if primary_hits == 0:
                    continue

                bonus_hits = self._count_matched_phrases(normalized_page, bonus_keywords)
                if bonus_keywords and bonus_hits < min_bonus_hits:
                    continue
                table_bonus = 0
                if any(token in normalized_page for token in ["table", "kpi", "unit", "單位", "单位", "關鍵績效", "关键绩效"]):
                    table_bonus += 1
                if any(token in normalized_page for token in ["2024", "2023", "2022", "二零二四年", "二零二三年"]):
                    table_bonus += 1

                score = primary_hits * 10 + bonus_hits * 3 + table_bonus * 2
                scored_pages.append(
                    {
                        "page_number": page_num,
                        "score": score,
                        "variable_name": variable_name,
                        "indicator_names": indicator_names,
                        "primary_hits": primary_hits,
                        "bonus_hits": bonus_hits,
                        "excerpt": re.sub(r"\s+", " ", page_text).strip()[:500],
                        "selection_reason": "variable_definition_page_match",
                    }
                )

            scored_pages.sort(key=lambda item: (-item["score"], item["page_number"]))
            for item in scored_pages[:max_candidates]:
                candidate_records.append(item)

        for item in sorted(candidate_records, key=lambda entry: (-entry["score"], entry["page_number"])):
            page_num = item["page_number"]
            if page_num not in selected_pages:
                selected_pages[page_num] = {
                    "page_number": page_num,
                    "score": item["score"],
                    "matched_variables": [item["variable_name"]],
                    "indicator_names": list(item["indicator_names"]),
                    "selection_reason": item["selection_reason"],
                    "excerpt": item["excerpt"],
                }
                continue

            existing = selected_pages[page_num]
            if item["variable_name"] not in existing["matched_variables"]:
                existing["matched_variables"].append(item["variable_name"])
            for indicator_name in item["indicator_names"]:
                if indicator_name not in existing["indicator_names"]:
                    existing["indicator_names"].append(indicator_name)
            existing["score"] = max(existing["score"], item["score"])

        page_to_indicators = {
            page_num: page_info["indicator_names"]
            for page_num, page_info in selected_pages.items()
        }

        return {
            "pages_to_process": sorted(selected_pages),
            "page_to_indicators": page_to_indicators,
            "detected_pages": [selected_pages[page_num] for page_num in sorted(selected_pages)],
        }

    def _load_keyword_results(self, keyword_results_dir: Path, pdf_name: str) -> Dict[str, Any]:
        result_file = keyword_results_dir / pdf_name / f"{pdf_name}_results.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Keyword results file not found for {pdf_name}: {result_file}")

        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _sanitize_text(self, text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"<[^>]+>", " ", lowered)
        lowered = re.sub(r"[\(\)\[\]{}“”\"'`]", " ", lowered)
        lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff%/&\-\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip()

    def _iter_chunks(self, chunks_payload: Any) -> List[Dict[str, Any]]:
        if not chunks_payload:
            return []

        if isinstance(chunks_payload, dict) and isinstance(chunks_payload.get("blocks"), list):
            return [item for item in chunks_payload["blocks"] if isinstance(item, dict)]

        if isinstance(chunks_payload, list):
            return [item for item in chunks_payload if isinstance(item, dict)]

        if isinstance(chunks_payload, dict) and any(
            key in chunks_payload for key in ("label", "block_type", "html", "content", "text", "bbox")
        ):
            return [chunks_payload]

        return []

    def _estimate_chunk_canvas_size(self, chunks: List[Dict[str, Any]]) -> tuple[float, float]:
        max_x = 0.0
        max_y = 0.0

        for chunk in chunks:
            bbox = chunk.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                _, _, x1, y1 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            max_x = max(max_x, x1)
            max_y = max(max_y, y1)

        return max_x, max_y

    def _normalize_bbox(
        self,
        bbox: Any,
        image_size: tuple[int, int],
        source_canvas_size: tuple[float, float] | None = None,
    ) -> List[int]:
        width, height = image_size

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return [0, 0, width, height]

        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return [0, 0, width, height]

        source_width, source_height = source_canvas_size or (width, height)
        if source_width > width or source_height > height:
            x_scale = width / source_width if source_width else 1.0
            y_scale = height / source_height if source_height else 1.0
            x0 *= x_scale
            x1 *= x_scale
            y0 *= y_scale
            y1 *= y_scale

        x0 = max(0, min(int(round(x0)), width))
        y0 = max(0, min(int(round(y0)), height))
        x1 = max(0, min(int(round(x1)), width))
        y1 = max(0, min(int(round(y1)), height))

        if x1 <= x0 or y1 <= y0:
            return [0, 0, width, height]

        return [x0, y0, x1, y1]

    def _chunk_content_text(self, chunk: Dict[str, Any]) -> str:
        for key in ("html", "content", "markdown", "text"):
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _extract_table_blocks(
        self,
        page_result: Dict[str, Any],
        pdf_name: str,
        page_num: int,
        image_file: Path,
    ) -> List[Dict[str, Any]]:
        table_blocks: List[Dict[str, Any]] = []
        table_crops_dir = self._get_temp_table_crops_dir(pdf_name)
        ensure_directory_exists(table_crops_dir)

        chunks = self._iter_chunks(page_result.get("chunks"))
        if not chunks:
            return table_blocks

        source_canvas_size = self._estimate_chunk_canvas_size(chunks)

        with Image.open(image_file) as img:
            image_size = img.size

            for block_index, chunk in enumerate(chunks, start=1):
                block_type = str(chunk.get("block_type") or chunk.get("label") or "")
                content_html = chunk.get("html") if isinstance(chunk.get("html"), str) else self._chunk_content_text(chunk)
                content_lower = content_html.lower() if isinstance(content_html, str) else ""

                is_table = block_type.lower() == "table" or "<table" in content_lower
                if not is_table:
                    continue

                bbox = self._normalize_bbox(
                    chunk.get("bbox"),
                    image_size,
                    source_canvas_size=source_canvas_size,
                )
                crop_path = table_crops_dir / f"page_{page_num:03d}_table_{block_index:02d}.png"
                try:
                    img.crop(tuple(bbox)).save(crop_path)
                except Exception:
                    crop_path = None

                table_blocks.append(
                    {
                        "table_block_id": chunk.get("id") or f"/page/{page_num}/Table/{block_index}",
                        "page_number": page_num,
                        "block_index": block_index,
                        "block_type": block_type,
                        "bbox": bbox,
                        "content_html": content_html,
                        "markdown": chunk.get("markdown"),
                        "text": chunk.get("text"),
                        "section_hierarchy": chunk.get("section_hierarchy", {}),
                        "metadata": chunk.get("metadata"),
                        "crop_image_path": str(crop_path) if crop_path else None,
                    }
                )

        return table_blocks

    def _run_page_ocr(self, image_path: Path) -> Dict[str, Any]:
        if self.ocr_backend == "datalab_api":
            return self._run_page_ocr_datalab(image_path)
        return self._run_page_ocr_chandra_local(image_path)

    def _ensure_chandra_manager(self):
        if self._chandra_manager is not None:
            return self._chandra_manager

        import os
        # Chandra reads MODEL_CHECKPOINT to find the local model path
        os.environ["MODEL_CHECKPOINT"] = str(Config.CHANDRA_MODEL_PATH)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            from chandra.model import InferenceManager
        except ImportError as exc:
            raise ImportError(
                "chandra-ocr is not installed. Install it with: pip install 'chandra-ocr[hf]'"
            ) from exc

        logger.info(f"Loading local Chandra model from {Config.CHANDRA_MODEL_PATH}; first load can take ~1 minute")
        self._chandra_manager = InferenceManager(method="hf")
        return self._chandra_manager

    def _run_page_ocr_chandra_local(self, image_path: Path) -> Dict[str, Any]:
        from chandra.model.schema import BatchInputItem

        manager = self._ensure_chandra_manager()

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        item = BatchInputItem(image=image, prompt_type="ocr_layout")
        output = manager.generate([item])[0]

        return {
            "markdown": output.markdown or "",
            "html": output.html or "",
            # chandra chunks: list of {bbox:[x0,y0,x1,y1] px, label:"Table"/"Text"/..., content:html}
            "chunks": output.chunks or [],
            "status": "complete" if not output.error else "failed",
            "error": "chandra_generation_error" if output.error else None,
            "page_count": 1,
            "parse_quality_score": None,
            "runtime": None,
            "cost_breakdown": {},
            "versions": None,
            "checkpoint_id": None,
            "output_format": "markdown,html,chunks",
        }

    def _run_page_ocr_batch(
        self,
        image_paths: List[Path],
        timing: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Batch OCR entry point. Chandra backend does a single batched forward
        pass over all images (~3-5x throughput vs per-image calls); datalab
        backend falls back to sequential per-image calls.

        Parameters
        ----------
        image_paths:
            Images to OCR. Must all exist; caller pre-filters missing ones.
        timing:
            Optional dict that will be populated with timing breakdown
            (milliseconds): ``img_load_ms``, ``generate_wall_ms``,
            ``generate_gpu_ms``. Pass a fresh ``{}`` from the caller and read
            it back after the call returns.

        Returns one result dict per input image, in the same order.
        """
        if self.ocr_backend == "datalab_api":
            return [self._run_page_ocr_datalab(p) for p in image_paths]
        return self._run_page_ocr_batch_chandra_local(image_paths, timing=timing)

    def _run_page_ocr_batch_chandra_local(
        self,
        image_paths: List[Path],
        timing: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Run Chandra OCR on a batch of images in a single forward pass.

        Returns one result dict per input image (same schema as
        ``_run_page_ocr_chandra_local``). Caller is responsible for filtering
        out missing files before calling.

        If ``timing`` is provided, fills it with three keys (all milliseconds):
          * ``img_load_ms``        — PIL decode + BatchInputItem construction (CPU)
          * ``generate_wall_ms``   — wall-clock time inside ``manager.generate``
          * ``generate_gpu_ms``    — CUDA-event measured GPU kernel time for
                                     the same call. Diff vs wall = CPU overhead
                                     inside generate (tokenize / pad / decode).

        Raises GPUFatalError if CUDA error is detected, triggering immediate
        worker exit and GPU health marking.
        """
        import time as _time
        from chandra.model.schema import BatchInputItem
        from src.utils import GPUHealthTracker, GPUFatalError
        import torch

        manager = self._ensure_chandra_manager()

        # --- image loading (CPU) ---
        t_load0 = _time.perf_counter()
        items: List[BatchInputItem] = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            items.append(BatchInputItem(image=image, prompt_type="ocr_layout"))
        img_load_ms = (_time.perf_counter() - t_load0) * 1000

        # --- generate: measure wall + GPU kernel time ---
        # CUDA events give pure kernel time; diff vs wall clock is CPU
        # overhead (apply_chat_template / tokenize / pad / batch_decode).
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
        t_gen0 = _time.perf_counter()

        # --- CUDA error handling ---
        gpu_health = GPUHealthTracker()
        try:
            outputs = manager.generate(items)
        except RuntimeError as e:
            error_msg = str(e)
            # Check if this is a CUDA error
            if "CUDA" in error_msg or "cuda" in error_msg:
                # Get current GPU ID
                try:
                    gpu_id = torch.cuda.current_device()
                except Exception:
                    gpu_id = 0

                logger.error(f"Fatal CUDA error detected on GPU {gpu_id}: {error_msg}")

                # Mark GPU as unhealthy
                gpu_health.mark_gpu_unhealthy(gpu_id, error_msg)

                # Raise fatal error to trigger immediate worker exit
                raise GPUFatalError(
                    f"CUDA error on GPU {gpu_id}: {error_msg}. "
                    f"GPU marked as unhealthy. Worker exiting immediately."
                ) from e
            else:
                # Not a CUDA error, re-raise
                raise

        generate_wall_ms = (_time.perf_counter() - t_gen0) * 1000
        if use_cuda:
            end_event.record()
            end_event.synchronize()
            generate_gpu_ms = start_event.elapsed_time(end_event)

            # Check for CUDA errors that didn't raise exceptions
            try:
                torch.cuda.synchronize()
            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "cuda" in error_msg:
                    gpu_id = torch.cuda.current_device()
                    logger.error(f"Deferred CUDA error detected on GPU {gpu_id}: {error_msg}")
                    gpu_health.mark_gpu_unhealthy(gpu_id, error_msg)
                    raise GPUFatalError(
                        f"Deferred CUDA error on GPU {gpu_id}: {error_msg}. "
                        f"GPU marked as unhealthy. Worker exiting immediately."
                    ) from e
                else:
                    raise
        else:
            generate_gpu_ms = generate_wall_ms

        if timing is not None:
            timing["img_load_ms"] = img_load_ms
            timing["generate_wall_ms"] = generate_wall_ms
            timing["generate_gpu_ms"] = generate_gpu_ms

        return [
            {
                "markdown": output.markdown or "",
                "html": output.html or "",
                "chunks": output.chunks or [],
                "status": "complete" if not output.error else "failed",
                "error": "chandra_generation_error" if output.error else None,
                "page_count": 1,
                "token_count": getattr(output, "token_count", 0) or 0,
                "parse_quality_score": None,
                "runtime": None,
                "cost_breakdown": {},
                "versions": None,
                "checkpoint_id": None,
                "output_format": "markdown,html,chunks",
            }
            for output in outputs
        ]

    def _run_page_ocr_datalab(self, image_path: Path) -> Dict[str, Any]:
        client = self._ensure_client()

        try:
            from datalab_sdk import ConvertOptions
        except ImportError as exc:
            raise ImportError(
                "Datalab SDK is missing. Install it with: pip install -r requirements.txt"
            ) from exc

        options = ConvertOptions(
            output_format="markdown,html,chunks",
            mode=self.convert_mode,
            disable_image_extraction=True,
            add_block_ids=True,
            include_markdown_in_chunks=True,
            extras="table_row_bboxes,new_block_types",
        )

        result = client.convert(
            file_path=image_path,
            options=options,
            max_polls=Config.DATALAB_API_MAX_POLLS,
            poll_interval=Config.DATALAB_API_POLL_INTERVAL,
        )

        if not result.success:
            raise RuntimeError(result.error or f"Datalab conversion failed for {image_path.name}")

        return {
            "markdown": result.markdown or "",
            "html": result.html or "",
            "chunks": result.chunks or {},
            "status": result.status,
            "error": result.error,
            "page_count": result.page_count,
            "parse_quality_score": result.parse_quality_score,
            "runtime": result.runtime,
            "cost_breakdown": result.cost_breakdown or {},
            "versions": result.versions,
            "checkpoint_id": result.checkpoint_id,
            "output_format": result.output_format,
        }

    def _table_record_for_ocr_json(self, table_block: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "table_block_id": table_block.get("table_block_id"),
            "page_number": table_block.get("page_number"),
            "bbox": table_block.get("bbox"),
            "section_hierarchy": table_block.get("section_hierarchy", {}),
            "content_html": table_block.get("content_html"),
            "markdown": table_block.get("markdown"),
            "text": table_block.get("text"),
        }

    def _table_runtime_record(self, table_block: Dict[str, Any]) -> Dict[str, Any]:
        runtime_block = self._table_record_for_ocr_json(table_block)
        runtime_block["crop_image_path"] = table_block.get("crop_image_path")
        return runtime_block

    def _build_table_source_text(self, table_block: Dict[str, Any]) -> str:
        parts = [
            table_block.get("content_html") or "",
            table_block.get("markdown") or "",
            table_block.get("text") or "",
        ]
        return self._sanitize_text(" ".join(part for part in parts if isinstance(part, str)))

    # Unicode sub/superscript digits -> ASCII, so a unit like "tCO₂e" survives
    # sanitization (which would otherwise strip the ₂ to a space) and matches "tco2e".
    _SUB_SUPERSCRIPT_DIGITS = str.maketrans(
        "₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789"
    )

    def _normalize_unit_text(self, unit: Any) -> str:
        if unit is None:
            return ""
        # Map sub/superscript digits BEFORE sanitize, otherwise the ₂ in "tCO₂e" is
        # dropped before the co2 fix-ups can run.
        raw = str(unit).translate(self._SUB_SUPERSCRIPT_DIGITS)
        return self._sanitize_text(raw).replace("co2 e", "co2e").replace("co₂", "co2")

    def _normalize_value_text(self, value: Any) -> str:
        if value is None:
            return ""
        return self._sanitize_text(str(value))

    def _extract_definition(self, variable_name: str, variable_definitions: Dict[str, Any]) -> Dict[str, Any]:
        definition = variable_definitions.get(variable_name, {})
        return definition if isinstance(definition, dict) else {}

    def _contains_any_phrase(self, text: str, phrases: List[str]) -> bool:
        for phrase in phrases:
            normalized_phrase = self._sanitize_text(phrase)
            if normalized_phrase and normalized_phrase in text:
                return True
        return False

    def _count_matched_phrases(self, text: str, phrases: List[str]) -> int:
        match_count = 0
        for phrase in phrases:
            normalized_phrase = self._sanitize_text(phrase)
            if normalized_phrase and normalized_phrase in text:
                match_count += 1
        return match_count

    def _table_quantitative_signal_score(
        self,
        table_block: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
        variable_definitions: Dict[str, Any],
    ) -> tuple[int, Dict[str, Any]]:
        source_text = self._build_table_source_text(table_block)
        if not source_text:
            return 0, {
                "numeric_count": 0,
                "year_count": 0,
                "unit_hits": 0,
                "variable_hits": 0,
                "definition_hits": 0,
                "forbidden_hits": 0,
            }

        all_variables = sorted({variable for variables in indicator_variables_map.values() for variable in variables})
        numeric_count = len(re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", source_text))
        year_count = len(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", source_text))

        quantitative_units = [
            "tonne",
            "tonnes",
            "tons",
            "tco2e",
            "co2e",
            "kg",
            "mwh",
            "kwh",
            "m3",
            "%",
            "噸",
            "吨",
            "千克",
            "兆瓦時",
            "立方米",
            "百萬港元",
            "千港元",
            "萬元",
            "億元",
        ]
        variable_hits = 0
        for variable_name in all_variables:
            normalized_variable = self._sanitize_text(variable_name)
            if normalized_variable and normalized_variable in source_text:
                variable_hits += 1

        definition_hits = 0
        forbidden_hits = 0
        for definition in variable_definitions.values():
            if not isinstance(definition, dict):
                continue

            definition_hits += self._count_matched_phrases(
                source_text,
                definition.get("required_keywords", []) + definition.get("semantic_hints", []),
            )
            forbidden_hits += self._count_matched_phrases(
                source_text,
                definition.get("forbidden_keywords", []),
            )

        unit_hits = self._count_matched_phrases(source_text, quantitative_units)
        score = 0
        if numeric_count >= 2:
            score += 3
        if numeric_count >= 6:
            score += 2
        if year_count >= 1:
            score += 2
        if year_count >= 2:
            score += 2
        if unit_hits >= 1:
            score += 2
        if unit_hits >= 3:
            score += 2
        if variable_hits >= 1:
            score += 4
        if definition_hits >= 2:
            score += 3
        if definition_hits >= 5:
            score += 2

        # Penalize text-heavy stakeholder or narrative tables that are unlikely to produce quantitative KPI variables.
        text_only_penalty_terms = [
            "stakeholder",
            "持份者",
            "communication and responses",
            "溝通與回應",
            "probable issues of concern",
            "可能關注問題",
            "site visits",
            "meetings",
            "seminars",
            "interviews",
            "customers",
            "suppliers",
            "shareholders",
        ]
        if self._contains_any_phrase(source_text, text_only_penalty_terms) and numeric_count <= 1:
            score -= 6

        return score, {
            "numeric_count": numeric_count,
            "year_count": year_count,
            "unit_hits": unit_hits,
            "variable_hits": variable_hits,
            "definition_hits": definition_hits,
            "forbidden_hits": forbidden_hits,
        }

    def _should_analyze_table_with_qwen(
        self,
        table_block: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
        variable_definitions: Dict[str, Any],
        page_mode: str,
    ) -> tuple[bool, Dict[str, Any]]:
        score, diagnostics = self._table_quantitative_signal_score(
            table_block=table_block,
            indicator_variables_map=indicator_variables_map,
            variable_definitions=variable_definitions,
        )

        threshold = 4
        if page_mode == "keyword_page":
            threshold = 5
        elif page_mode == "variable_definition_page":
            threshold = 4
        elif page_mode == "table_scan":
            threshold = 3

        diagnostics["score"] = score
        diagnostics["threshold"] = threshold
        return score >= threshold, diagnostics

    def _unit_matches(self, normalized_unit: str, phrases: List[str]) -> bool:
        if not normalized_unit:
            return False
        return self._contains_any_phrase(normalized_unit, phrases)

    def _item_matches_variable_definition(
        self,
        item: Dict[str, Any],
        table_block: Dict[str, Any],
        variable_name: str,
        variable_definitions: Dict[str, Any],
        from_page_image: bool = False,
    ) -> bool:
        definition = self._extract_definition(variable_name, variable_definitions)
        if not definition:
            return True

        source_text = self._build_table_source_text(table_block)
        indicator_text = self._sanitize_text(str(item.get("indicator", "")))
        unit_text = self._normalize_unit_text(item.get("unit"))
        value_text = self._normalize_value_text(item.get("value"))

        source_requirement = definition.get("source_requirement")
        # The vision fallback reads a full PAGE IMAGE, legitimately not a table —
        # don't apply the table_only requirement to it (unit/%/forbidden + the
        # indicator-based required_keywords check below still guard it).
        if source_requirement == "table_only" and not source_text and not from_page_image:
            return False

        required_keywords = definition.get("required_keywords", [])
        if required_keywords:
            # Satisfy required keywords from the table OR the matched row label, with
            # scope-synonym normalization: tables often write "Direct greenhouse gas
            # emissions" (no literal "Scope 1"/adjacent "direct emission"), which the
            # source-only check rejected even though the matcher already confirmed the
            # row's scope (2903499: 7B nailed Scope 1/2/3, all killed here).
            indicator_canon = self._inject_scope_synonyms(indicator_text)
            if not (
                self._contains_any_phrase(source_text, required_keywords)
                or self._contains_any_phrase(indicator_text, required_keywords)
                or self._contains_any_phrase(indicator_canon, required_keywords)
            ):
                return False

        semantic_hints = definition.get("semantic_hints", [])
        if semantic_hints:
            semantic_matches = self._count_matched_phrases(source_text, semantic_hints)
            # semantic_hints is a TABLE-CONTEXT check (e.g. "total"/"scope 2" must
            # appear near the value). The page-image fallback has no surrounding
            # table text — source_text is empty — so a clean indicator like
            # "greenhouse gas emission" that already passed required_keywords + a
            # high match score would be wrongly killed here. When from_page_image
            # and there is no source_text, don't let an unmet hint veto it.
            indicator_has_hint = self._contains_any_phrase(indicator_text, semantic_hints)
            if semantic_matches == 0 and not indicator_has_hint:
                if not (from_page_image and not source_text):
                    return False

        forbidden_keywords = definition.get("forbidden_keywords", [])
        if forbidden_keywords and self._contains_any_phrase(source_text, forbidden_keywords):
            indicator_required = definition.get("allow_forbidden_if_indicator_matches", False)
            if not indicator_required or not self._contains_any_phrase(indicator_text, semantic_hints):
                return False

        # forbidden_units stays a HARD filter: it carries the real disambiguation
        # (kWh / m3 / kg etc. → energy/water/volume, not GHG mass).
        forbidden_units = definition.get("forbidden_units", [])
        if forbidden_units and self._unit_matches(unit_text, forbidden_units):
            return False

        # allowed_units is a SOFT signal, NOT a kill-switch. A row whose scope label
        # matched strongly must not be silently dropped just because its unit string
        # didn't match the whitelist after normalization (e.g. the tCO₂e subscript
        # regression that nuked 1345). forbidden_units above already excludes the real
        # confusables, so an unrecognized-but-not-forbidden unit is allowed through.

        value_patterns = definition.get("value_patterns", [])
        if value_patterns:
            if not any(re.search(pattern, value_text, re.IGNORECASE) for pattern in value_patterns):
                return False

        # min_confidence guards table extraction, where 7B emits a per-row confidence.
        # The page-image vision fallback prompt does not ask for confidence, so its
        # items legitimately have none — applying this filter would kill every
        # vision-recovered scope row (that was the `added 0` bug). The match score +
        # required_keywords + forbidden_units checks above already guard these.
        min_confidence = definition.get("min_confidence")
        if min_confidence is not None and not from_page_image:
            try:
                if float(item.get("confidence", 0)) < float(min_confidence):
                    return False
            except (TypeError, ValueError):
                return False

        return True

    def _derive_total_scope_1_2_if_missing(
        self,
        results: Dict[str, Any],
        indicator_name: str,
        variable_name: str,
    ) -> None:
        if variable_name != "greenhouse gas emission":
            return

        indicator_bucket = results["all_variables_collected"].get(indicator_name, {})
        total_items = indicator_bucket.get("greenhouse gas emission", [])
        if total_items:
            return

        scope1_items = indicator_bucket.get("Scope 1 (direct emissions)", [])
        scope2_items = indicator_bucket.get("Scope 2 (indirect emissions)", [])
        if not scope1_items or not scope2_items:
            return

        scope1_by_year = {item.get("year"): item for item in scope1_items if item.get("year") is not None}
        scope2_by_year = {item.get("year"): item for item in scope2_items if item.get("year") is not None}

        for year in sorted(set(scope1_by_year) & set(scope2_by_year)):
            left = scope1_by_year[year]
            right = scope2_by_year[year]
            try:
                total_value = float(left.get("value")) + float(right.get("value"))
            except (TypeError, ValueError):
                continue

            left_unit = str(left.get("unit") or "").strip()
            right_unit = str(right.get("unit") or "").strip()
            unit = left_unit or right_unit
            if left_unit and right_unit and left_unit != right_unit:
                continue

            derived_item = {
                "indicator": "greenhouse gas emission",
                "value": round(total_value, 6),
                "unit": unit,
                "raw_year": left.get("raw_year") or right.get("raw_year"),
                "confidence": min(float(left.get("confidence", 0.8)), float(right.get("confidence", 0.8))),
                "source_page": left.get("source_page") or right.get("source_page"),
                "year": year,
                "year_source": year,
                "source_type": "table",
                "traceability": {
                    "page_number": left.get("source_page") or right.get("source_page"),
                    "source_format": "table_or_chart",
                    "derivation": "scope_1_plus_scope_2",
                    "source_variables": [
                        "Scope 1 (direct emissions)",
                        "Scope 2 (indirect emissions)",
                    ],
                },
            }
            self.image_recognizer._merge_indicator_item(
                results,
                indicator_name,
                "greenhouse gas emission",
                derived_item,
            )

    # Labels that mark a row as a subtotal/total rather than a leaf component.
    # If one appears in a group we trust it instead of re-summing the components.
    _COMPONENT_SUBTOTAL_TERMS = (
        "total", "subtotal", "合计", "合計", "小计", "小計", "总计", "總計", "总额", "總額",
    )

    def _aggregate_table_components(
        self,
        results: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
    ) -> None:
        """Sum split component rows of the same scope/year within one table.

        Chandra splits a rowspan'd scope label (e.g. a "Scope 1" cell spanning the
        Petrol and Diesel rows) into separate rows, so each component row is
        extracted as its own item that still matches the same scope variable
        (Petrol -> Scope 1, Diesel -> Scope 1). Downstream consumers and
        ``gt_compare.py`` expect one aggregate value per scope/year, but the
        collector keeps the components side by side and "last write wins" keeps
        only one of them (1082: Scope 1 came out as 46.12 / Petrol only, vs
        GT 54.89 = Petrol + Diesel). Replace each same-table, same-variable,
        same-year component group with a single summed item.
        """
        for indicator_name, variable_names in indicator_variables_map.items():
            bucket = results["all_variables_collected"].get(indicator_name, {})
            for variable_name in variable_names:
                items = bucket.get(variable_name, [])
                if len(items) < 2:
                    continue

                grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
                passthrough: List[Dict[str, Any]] = []
                for item in items:
                    table_block_id = (item.get("traceability") or {}).get("table_block_id")
                    year = self.image_recognizer._coerce_optional_year(item.get("year"))
                    if item.get("source_type") != "table" or table_block_id is None or year is None:
                        passthrough.append(item)
                        continue
                    grouped.setdefault((table_block_id, year), []).append(item)

                rebuilt: List[Dict[str, Any]] = list(passthrough)
                changed = False
                for group in grouped.values():
                    aggregated = self._sum_component_group(group)
                    if aggregated is not group:
                        changed = True
                    rebuilt.extend(aggregated)

                if changed:
                    bucket[variable_name] = rebuilt

    def _sum_component_group(self, group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a one-item summed group, or the original group if summing is unsafe.

        Conservative on purpose — abort (return the group unchanged) when a
        subtotal/total row is present, when any value is non-numeric, or when units
        are mixed. Only distinct, labeled component rows are summed; rows with no
        label or a duplicate (label, value) are skipped so a re-extracted row is
        never double counted.
        """
        if len(group) < 2:
            return group

        components: List[Tuple[Dict[str, Any], float]] = []
        seen: Set[Tuple[str, float]] = set()
        unit_text: Optional[str] = None

        for item in group:
            label = self._sanitize_text(str(item.get("component_label", "")))
            if any(term in label for term in self._COMPONENT_SUBTOTAL_TERMS):
                return group
            value = self._coerce_numeric_value(item.get("value"))
            if value is None:
                return group
            item_unit = self._normalize_unit_text(item.get("unit"))
            if item_unit:
                if unit_text is None:
                    unit_text = item_unit
                elif item_unit != unit_text:
                    return group

            if not label:
                continue
            signature = (label, round(value, 6))
            if signature in seen:
                continue
            seen.add(signature)
            components.append((item, value))

        if len(components) < 2:
            return group

        total = sum(value for _, value in components)
        base = max(components, key=lambda cv: self._item_confidence(cv[0]))[0].copy()
        base["value"] = self._normalize_derived_value(total)
        base["component_aggregation"] = {
            "method": "component_sum",
            "components": [
                {"label": item.get("component_label"), "value": value}
                for item, value in components
            ],
        }
        return [base]

    def _coerce_numeric_value(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if value is None:
            return None

        text = str(value).strip()
        if not text or text.lower() in {"n/a", "na", "none", "null", "not applicable", "not available"}:
            return None

        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text)
        if not match:
            return None

        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

    def _normalize_derived_value(self, value: float) -> int | float:
        rounded = round(value, 6)
        if abs(rounded - round(rounded)) < 1e-9:
            return int(round(rounded))
        return rounded

    def _item_confidence(self, item: Dict[str, Any] | None) -> float:
        if not item:
            return 0.8
        try:
            return float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            return 0.8

    def _numeric_items_by_year(self, items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        items_by_year: Dict[int, Dict[str, Any]] = {}

        for item in items:
            year = self.image_recognizer._coerce_optional_year(item.get("year"))
            value = self._coerce_numeric_value(item.get("value"))
            if year is None or value is None:
                continue

            existing = items_by_year.get(year)
            if existing is None or self._item_confidence(item) > self._item_confidence(existing):
                items_by_year[year] = item

        return items_by_year

    def _first_formula_for_target(self, formulas: List[str], target_variable_name: str) -> str:
        target_normalized = self._sanitize_text(target_variable_name)

        for formula in formulas:
            if "=" not in formula:
                continue
            left_side = formula.split("=", 1)[0]
            left_normalized = self._sanitize_text(left_side)
            if left_normalized and (left_normalized in target_normalized or target_normalized in left_normalized):
                return formula

        return formulas[0] if formulas else ""

    def _normalize_formula_text(self, text: str) -> str:
        normalized = str(text or "").lower()
        normalized = normalized.replace("（", "(").replace("）", ")")
        normalized = normalized.replace("，", ",").replace("：", ":")
        normalized = normalized.replace("＋", "+").replace("－", "-")
        normalized = normalized.replace("×", "*").replace("÷", "/")
        normalized = normalized.replace("scope1", "scope 1").replace("scope2", "scope 2").replace("scope3", "scope 3")
        normalized = re.sub(r"\s*&\s*", "&", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _build_formula_alias_map(self, variable_names: List[str]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}

        def register(alias: str, variable_name: str) -> None:
            normalized_alias = self._normalize_formula_text(alias)
            if normalized_alias:
                alias_map.setdefault(normalized_alias, variable_name)

        for variable_name in variable_names:
            register(variable_name, variable_name)
            register(self._sanitize_text(variable_name), variable_name)

        if "greenhouse gas emission" in variable_names:
            register(
                "greenhouse gas emission of scope 1 (direct emissions) and scope 2 (indirect emissions)",
                "greenhouse gas emission",
            )
            register(
                "greenhouse gas emission of scope 1 and scope 2",
                "greenhouse gas emission",
            )

        if "emission intensity of Scope1&2" in variable_names:
            register("emission intensity of scope 1&2", "emission intensity of Scope1&2")
            register("emission intensity of scope 1 and scope 2", "emission intensity of Scope1&2")

        if "benchmark score in Scope1&2" in variable_names:
            register("benchmark score in scope 1&2", "benchmark score in Scope1&2")

        if "industry average emission intensity value of Scope 1&2" in variable_names:
            register(
                "industry average emission intensity value of scope1&2",
                "industry average emission intensity value of Scope 1&2",
            )
            register(
                "industry average emission intensity value of scope 1&2",
                "industry average emission intensity value of Scope 1&2",
            )

        if "emission intensity of Scope3" in variable_names:
            register("emission intensity of scope 3", "emission intensity of Scope3")

        if "benchmark score in Scope3" in variable_names:
            register("benchmark score in scope 3", "benchmark score in Scope3")

        return alias_map

    def _parse_formula_definition(
        self,
        formula_text: str,
        variable_names: List[str],
    ) -> Optional[Dict[str, Any]]:
        if "=" not in formula_text:
            return None

        left_text, right_text = formula_text.split("=", 1)
        target_raw = left_text.strip()
        expression_raw = right_text.strip()
        normalized_expression = self._normalize_formula_text(expression_raw)

        absolute_value = False
        if "get absoluate value" in normalized_expression or "get absolute value" in normalized_expression:
            absolute_value = True
            normalized_expression = normalized_expression.replace("get absoluate value", "")
            normalized_expression = normalized_expression.replace("get absolute value", "")
            normalized_expression = normalized_expression.replace(",", " ")
            normalized_expression = re.sub(r"\s+", " ", normalized_expression).strip()

        alias_map = self._build_formula_alias_map(variable_names + [target_raw])
        normalized_target = self._normalize_formula_text(target_raw)
        target_variable_name = alias_map.get(normalized_target, target_raw)

        reference_specs: List[Dict[str, Any]] = []
        parsed_expression = normalized_expression
        token_index = 0

        replacement_candidates: List[Tuple[str, str, str]] = []
        for alias_text, variable_name in alias_map.items():
            replacement_candidates.append((f"this year {alias_text}", variable_name, "current"))
            replacement_candidates.append((f"last year {alias_text}", variable_name, "previous"))
            replacement_candidates.append((alias_text, variable_name, "same"))

        replacement_candidates.sort(key=lambda item: len(item[0]), reverse=True)

        for phrase_text, variable_name, year_mode in replacement_candidates:
            pattern = re.compile(
                rf"(?<![a-z0-9\u4e00-\u9fff_]){re.escape(phrase_text)}(?![a-z0-9\u4e00-\u9fff_])"
            )
            while True:
                match = pattern.search(parsed_expression)
                if not match:
                    break
                token = f"__var_{token_index}__"
                token_index += 1
                reference_specs.append(
                    {
                        "token": token,
                        "variable_name": variable_name,
                        "year_mode": year_mode,
                    }
                )
                parsed_expression = parsed_expression[:match.start()] + token + parsed_expression[match.end():]

        parsed_expression = re.sub(r"\s+", " ", parsed_expression).strip()
        if not parsed_expression:
            return None

        return {
            "formula_text": formula_text,
            "target_variable_name": target_variable_name,
            "expression": parsed_expression,
            "references": reference_specs,
            "absolute_value": absolute_value,
        }

    def _safe_eval_formula_expression(self, expression: str, token_values: Dict[str, float]) -> Optional[float]:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Num):
                return float(node.n)
            if isinstance(node, ast.Name):
                if node.id not in token_values:
                    raise KeyError(node.id)
                return float(token_values[node.id])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left = evaluate(node.left)
                right = evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        raise ZeroDivisionError()
                    return left / right
            raise ValueError(f"Unsupported formula node: {ast.dump(node)}")

        try:
            return evaluate(tree)
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            return None

    def _existing_years_for_variable(
        self,
        results: Dict[str, Any],
        indicator_name: str,
        variable_name: str,
    ) -> Set[int]:
        existing_items = results["all_variables_collected"].get(indicator_name, {}).get(variable_name, [])
        return {
            year
            for year in (
                self.image_recognizer._coerce_optional_year(item.get("year"))
                for item in existing_items
            )
            if year is not None
        }

    def _infer_formula_result_unit(
        self,
        expression: str,
        referenced_items: List[Dict[str, Any]],
    ) -> str:
        units = [str(item.get("unit") or "").strip() for item in referenced_items if item]
        non_empty_units = [unit for unit in units if unit]
        if not non_empty_units:
            return ""

        if "/" in expression:
            if len(non_empty_units) >= 2:
                return self._derive_ratio_unit(non_empty_units[0], non_empty_units[1], "value")
            return "ratio"

        if all(unit == non_empty_units[0] for unit in non_empty_units):
            return non_empty_units[0]

        percent_like_units = {unit for unit in non_empty_units if unit in {"%", "ratio", "score"}}
        if percent_like_units and len(percent_like_units) == len(set(non_empty_units)):
            return "%"

        return ""

    def _apply_generic_formula_completion(
        self,
        results: Dict[str, Any],
        indicator_name: str,
        variable_names: List[str],
        formulas: List[str],
    ) -> int:
        if not formulas:
            return 0

        indicator_bucket = results["all_variables_collected"].get(indicator_name, {})
        numeric_items_by_variable = {
            variable_name: self._numeric_items_by_year(indicator_bucket.get(variable_name, []))
            for variable_name in list(indicator_bucket.keys()) + list(variable_names)
        }
        created_count = 0

        for formula_text in formulas:
            parsed_formula = self._parse_formula_definition(formula_text, variable_names)
            if not parsed_formula:
                continue

            target_variable_name = parsed_formula["target_variable_name"]
            references = parsed_formula["references"]
            if not references:
                continue

            candidate_years: Set[int] = set()
            for reference in references:
                for year in numeric_items_by_variable.get(reference["variable_name"], {}).keys():
                    if reference["year_mode"] == "previous":
                        candidate_years.add(year + 1)
                    else:
                        candidate_years.add(year)

            existing_years = self._existing_years_for_variable(results, indicator_name, target_variable_name)
            formula_created_for_target = 0

            for year in sorted(candidate_years):
                if year in existing_years:
                    continue

                token_values: Dict[str, float] = {}
                referenced_items: List[Dict[str, Any]] = []
                referenced_variable_names: List[str] = []
                missing_dependency = False

                for reference in references:
                    source_year = year - 1 if reference["year_mode"] == "previous" else year
                    source_item = numeric_items_by_variable.get(reference["variable_name"], {}).get(source_year)
                    if source_item is None:
                        missing_dependency = True
                        break

                    numeric_value = self._coerce_numeric_value(source_item.get("value"))
                    if numeric_value is None:
                        missing_dependency = True
                        break

                    token_values[reference["token"]] = numeric_value
                    referenced_items.append(source_item)
                    referenced_variable_names.append(reference["variable_name"])

                if missing_dependency:
                    continue

                derived_value = self._safe_eval_formula_expression(parsed_formula["expression"], token_values)
                if derived_value is None:
                    continue
                if parsed_formula["absolute_value"]:
                    derived_value = abs(derived_value)

                derived_item = self._make_formula_item(
                    variable_name=target_variable_name,
                    value=derived_value,
                    unit=self._infer_formula_result_unit(parsed_formula["expression"], referenced_items),
                    year=year,
                    source_items=referenced_items,
                    derivation="generic_formula_evaluation",
                    formula_text=parsed_formula["formula_text"],
                    source_variables=sorted(set(referenced_variable_names)),
                )
                self.image_recognizer._merge_indicator_item(
                    results,
                    indicator_name,
                    target_variable_name,
                    derived_item,
                )
                existing_years.add(year)
                formula_created_for_target += 1
                created_count += 1

            if formula_created_for_target > 0:
                results["enhancement_metadata"].setdefault("generic_formula_completions", []).append(
                    {
                        "indicator": indicator_name,
                        "target_variable": target_variable_name,
                        "formula": formula_text,
                        "created_items": formula_created_for_target,
                    }
                )

        return created_count

    def _make_formula_item(
        self,
        variable_name: str,
        value: float,
        unit: str,
        year: int | None,
        source_items: List[Dict[str, Any]],
        derivation: str,
        formula_text: str,
        source_variables: List[str],
    ) -> Dict[str, Any]:
        source_pages = [
            item.get("source_page")
            for item in source_items
            if isinstance(item, dict) and item.get("source_page") is not None
        ]
        source_pages = sorted({int(page) for page in source_pages if isinstance(page, int) or str(page).isdigit()})
        confidence_candidates = [self._item_confidence(item) for item in source_items if item]
        confidence = min(confidence_candidates) if confidence_candidates else 0.8

        return {
            "indicator": variable_name,
            "value": self._normalize_derived_value(value),
            "unit": unit,
            "raw_year": str(year) if year is not None else None,
            "confidence": round(confidence, 4),
            "source_page": source_pages[0] if source_pages else None,
            "year": year,
            "year_source": year,
            "source_type": "formula",
            "traceability": {
                "page_number": source_pages[0] if source_pages else None,
                "source_format": "formula_derivation",
                "derivation": derivation,
                "formula": formula_text,
                "source_variables": source_variables,
                "source_pages": source_pages,
            },
        }

    def _source_priority(self, item: Dict[str, Any]) -> int:
        source_type = str(item.get("source_type") or "").lower()
        if source_type == "table":
            return 3
        if source_type == "formula":
            return 2
        if source_type == "text":
            return 1
        return 0

    def _candidate_sort_key(self, item: Dict[str, Any]) -> tuple:
        year = self.image_recognizer._coerce_optional_year(item.get("year")) or -1
        confidence = self._item_confidence(item)
        has_value = 1 if self.image_recognizer._has_substantive_value(item) else 0
        source_priority = self._source_priority(item)
        page_num = item.get("source_page")
        try:
            page_num = int(page_num)
        except (TypeError, ValueError):
            page_num = 10**9

        return (
            has_value,
            source_priority,
            confidence,
            year,
            -page_num,
        )

    def _sort_and_trim_indicator_variables(
        self,
        results: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
    ) -> None:
        ranking_summary: List[Dict[str, Any]] = []

        for indicator_name, variable_names in indicator_variables_map.items():
            indicator_bucket = results["all_variables_collected"].get(indicator_name, {})

            for variable_name in variable_names:
                items = indicator_bucket.get(variable_name, [])
                if not items:
                    continue

                deduped_items: List[Dict[str, Any]] = []
                for item in items:
                    if any(self.image_recognizer._is_exact_duplicate_item(item, existing) for existing in deduped_items):
                        continue
                    deduped_items.append(item)

                substantive_items = [
                    item for item in deduped_items
                    if self.image_recognizer._has_substantive_value(item)
                ]
                missing_items = [
                    item for item in deduped_items
                    if not self.image_recognizer._has_substantive_value(item)
                ]

                substantive_items.sort(key=self._candidate_sort_key, reverse=True)
                missing_items.sort(key=self._candidate_sort_key, reverse=True)

                # Keep the best item per distinct year (one value per scope per year),
                # not a flat top-N, so a 3+ year table keeps every year column. Items
                # are pre-sorted best-first, so the first hit for a year is its best.
                trimmed_items: List[Dict[str, Any]] = []
                seen_years: Set[Any] = set()
                kept_unyeared = False
                for item in substantive_items:
                    year = self.image_recognizer._coerce_optional_year(item.get("year"))
                    if year is None:
                        if kept_unyeared:
                            continue
                        kept_unyeared = True
                    elif year in seen_years:
                        continue
                    else:
                        seen_years.add(year)
                    trimmed_items.append(item)
                    if len(trimmed_items) >= self.MAX_YEARS_PER_VARIABLE:
                        break
                if not trimmed_items and missing_items:
                    trimmed_items = missing_items[:1]

                if len(trimmed_items) != len(items) or trimmed_items != items:
                    ranking_summary.append(
                        {
                            "indicator": indicator_name,
                            "variable": variable_name,
                            "original_count": len(items),
                            "final_count": len(trimmed_items),
                        }
                    )

                indicator_bucket[variable_name] = trimmed_items

        results["enhancement_metadata"]["variable_ranking"] = {
            "enabled": True,
            "max_years_per_variable": self.MAX_YEARS_PER_VARIABLE,
            "priority_order": ["table", "formula", "text"],
            "adjusted_variables": ranking_summary,
        }

    def _derive_ratio_unit(
        self,
        numerator_unit: str | None,
        denominator_unit: str | None,
        fallback_denominator_label: str,
    ) -> str:
        numerator = str(numerator_unit or "").strip()
        denominator = str(denominator_unit or "").strip()

        if numerator and denominator:
            return f"{numerator}/{denominator}"
        if numerator:
            return f"{numerator}/{fallback_denominator_label}"
        return "ratio"

    def _derive_denominator_unit_from_ratio(
        self,
        ratio_unit: str | None,
        default_unit: str,
    ) -> str:
        unit_text = str(ratio_unit or "").strip()
        if "/" in unit_text:
            return unit_text.split("/", 1)[1].strip() or default_unit
        if " per " in unit_text.lower():
            return unit_text.lower().split(" per ", 1)[1].strip() or default_unit
        return default_unit

    def _derive_numerator_unit_from_ratio(
        self,
        ratio_unit: str | None,
        default_unit: str,
    ) -> str:
        unit_text = str(ratio_unit or "").strip()
        if "/" in unit_text:
            return unit_text.split("/", 1)[0].strip() or default_unit
        if " per " in unit_text.lower():
            return unit_text.lower().split(" per ", 1)[0].strip() or default_unit
        return default_unit

    def _ratio_item_supports_revenue_derivation(self, ratio_item: Dict[str, Any] | None) -> bool:
        if not ratio_item:
            return False

        combined_text = " ".join(
            [
                str(ratio_item.get("indicator") or ""),
                str(ratio_item.get("unit") or ""),
                str(((ratio_item.get("traceability") or {}).get("table_excerpt")) or ""),
            ]
        )
        normalized = self._sanitize_text(combined_text)

        revenue_markers = [
            "revenue",
            "turnover",
            "income",
            "收益",
            "收入",
            "营业收入",
            "營業收入",
        ]
        employee_markers = [
            "employee",
            "employees",
            "staff",
            "僱員",
            "员工",
            "每名僱員",
            "每名员工",
        ]

        if any(self._sanitize_text(marker) in normalized for marker in employee_markers):
            return False

        if any(self._sanitize_text(marker) in normalized for marker in revenue_markers):
            return True

        unit_text = str(ratio_item.get("unit") or "").lower()
        if any(marker in unit_text for marker in ["employee", "staff"]):
            return False

        return False

    def _derive_division_relationship(
        self,
        results: Dict[str, Any],
        indicator_name: str,
        formulas: List[str],
        numerator_variable: str,
        denominator_variable: str,
        ratio_variable: str,
        ratio_scale: float = 1.0,
        default_denominator_unit: str = "currency",
        default_numerator_unit: str = "",
    ) -> None:
        indicator_bucket = results["all_variables_collected"].get(indicator_name, {})

        numerator_by_year = self._numeric_items_by_year(indicator_bucket.get(numerator_variable, []))
        denominator_by_year = self._numeric_items_by_year(indicator_bucket.get(denominator_variable, []))
        ratio_by_year = self._numeric_items_by_year(indicator_bucket.get(ratio_variable, []))
        all_years = sorted(set(numerator_by_year) | set(denominator_by_year) | set(ratio_by_year))

        ratio_formula = self._first_formula_for_target(formulas, ratio_variable)

        for year in all_years:
            numerator_item = numerator_by_year.get(year)
            denominator_item = denominator_by_year.get(year)
            ratio_item = ratio_by_year.get(year)

            numerator_value = self._coerce_numeric_value(numerator_item.get("value")) if numerator_item else None
            denominator_value = self._coerce_numeric_value(denominator_item.get("value")) if denominator_item else None
            ratio_value = self._coerce_numeric_value(ratio_item.get("value")) if ratio_item else None

            if numerator_value is not None and denominator_value not in (None, 0) and ratio_item is None:
                derived_ratio = (numerator_value / denominator_value) * ratio_scale
                derived_item = self._make_formula_item(
                    variable_name=ratio_variable,
                    value=derived_ratio,
                    unit=self._derive_ratio_unit(
                        numerator_item.get("unit") if numerator_item else "",
                        denominator_item.get("unit") if denominator_item else "",
                        fallback_denominator_label=denominator_variable,
                    ),
                    year=year,
                    source_items=[numerator_item, denominator_item],
                    derivation="division_ratio",
                    formula_text=ratio_formula,
                    source_variables=[numerator_variable, denominator_variable],
                )
                self.image_recognizer._merge_indicator_item(results, indicator_name, ratio_variable, derived_item)

            if numerator_value is not None and ratio_value not in (None, 0) and denominator_item is None:
                if denominator_variable == "revenue" and not self._ratio_item_supports_revenue_derivation(ratio_item):
                    continue
                derived_denominator = numerator_value / (ratio_value / ratio_scale)
                derived_item = self._make_formula_item(
                    variable_name=denominator_variable,
                    value=derived_denominator,
                    unit=self._derive_denominator_unit_from_ratio(
                        ratio_item.get("unit") if ratio_item else "",
                        default_unit=default_denominator_unit,
                    ),
                    year=year,
                    source_items=[numerator_item, ratio_item],
                    derivation="rearranged_division_denominator",
                    formula_text=ratio_formula,
                    source_variables=[numerator_variable, ratio_variable],
                )
                self.image_recognizer._merge_indicator_item(results, indicator_name, denominator_variable, derived_item)

            if denominator_value is not None and ratio_value is not None and numerator_item is None:
                derived_numerator = denominator_value * (ratio_value / ratio_scale)
                derived_item = self._make_formula_item(
                    variable_name=numerator_variable,
                    value=derived_numerator,
                    unit=self._derive_numerator_unit_from_ratio(
                        ratio_item.get("unit") if ratio_item else "",
                        default_unit=default_numerator_unit,
                    ),
                    year=year,
                    source_items=[denominator_item, ratio_item],
                    derivation="rearranged_division_numerator",
                    formula_text=ratio_formula,
                    source_variables=[denominator_variable, ratio_variable],
                )
                self.image_recognizer._merge_indicator_item(results, indicator_name, numerator_variable, derived_item)

    def _derive_improvement_rate(
        self,
        results: Dict[str, Any],
        indicator_name: str,
        formulas: List[str],
        base_variable: str,
        target_variable: str,
    ) -> None:
        indicator_bucket = results["all_variables_collected"].get(indicator_name, {})
        base_by_year = self._numeric_items_by_year(indicator_bucket.get(base_variable, []))
        if len(base_by_year) < 2:
            return

        target_formula = self._first_formula_for_target(formulas, target_variable)
        years = sorted(base_by_year)

        for index in range(1, len(years)):
            previous_year = years[index - 1]
            current_year = years[index]
            previous_item = base_by_year[previous_year]
            current_item = base_by_year[current_year]
            previous_value = self._coerce_numeric_value(previous_item.get("value"))
            current_value = self._coerce_numeric_value(current_item.get("value"))

            if previous_value in (None, 0) or current_value is None:
                continue

            derived_value = abs((current_value - previous_value) / previous_value) * 100
            derived_item = self._make_formula_item(
                variable_name=target_variable,
                value=derived_value,
                unit="%",
                year=current_year,
                source_items=[previous_item, current_item],
                derivation="year_over_year_improvement_rate",
                formula_text=target_formula,
                source_variables=[base_variable],
            )
            derived_item["raw_year"] = f"{previous_year}->{current_year}"
            self.image_recognizer._merge_indicator_item(results, indicator_name, target_variable, derived_item)

    def _count_scope3_categories_from_table(self, table_block: Dict[str, Any]) -> int:
        markdown = table_block.get("markdown") or ""
        if not markdown:
            return 0

        lines = [line.strip() for line in markdown.splitlines() if line.strip().startswith("|")]
        in_scope3 = False
        category_count = 0

        for line in lines:
            line_plain = re.sub(r"<[^>]+>", " ", line)
            normalized = self._sanitize_text(line_plain)
            if not normalized:
                continue

            if any(term in normalized for term in ["scope 3", "scope3", "範圍3", "范围3"]):
                in_scope3 = True
                continue

            if not in_scope3:
                continue

            if any(term in normalized for term in ["total", "總計", "总计"]):
                break

            if any(term in normalized for term in ["scope 1", "scope1", "範圍1", "范围1", "scope 2", "scope2", "範圍2", "范围2"]):
                break

            if re.search(r"-{3,}", line_plain):
                continue

            has_number = bool(re.search(r"-?\d[\d,]*(?:\.\d+)?", line_plain))
            has_label = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", line_plain))
            if has_number and has_label:
                category_count += 1

        return category_count

    def _derive_scope3_category_metrics(
        self,
        results: Dict[str, Any],
        indicator_name: str,
        formulas: List[str],
        all_tables: List[Dict[str, Any]],
        reporting_period_context: Dict[str, Any],
    ) -> None:
        indicator_bucket = results["all_variables_collected"].get(indicator_name, {})
        count_variable = "number of Scope 3 category disclosed"
        coverage_variable = "coverage rate of Scope 3"
        default_year = self.image_recognizer._coerce_optional_year(reporting_period_context.get("default_year"))

        count_items_by_year = self._numeric_items_by_year(indicator_bucket.get(count_variable, []))
        if not count_items_by_year:
            best_count = 0
            best_table = None

            for table_block in all_tables:
                table_text = self._build_table_source_text(table_block)
                if not any(term in table_text for term in ["scope 3", "scope3", "範圍3", "范围3"]):
                    continue

                category_count = self._count_scope3_categories_from_table(table_block)
                if category_count > best_count:
                    best_count = category_count
                    best_table = table_block

            if best_table is not None and best_count > 0:
                derived_count_item = {
                    "indicator": count_variable,
                    "value": best_count,
                    "unit": "category",
                    "raw_year": str(default_year) if default_year is not None else None,
                    "confidence": 0.8,
                    "source_page": best_table.get("page_number"),
                    "year": default_year,
                    "year_source": default_year,
                    "source_type": "formula",
                    "traceability": {
                        "page_number": best_table.get("page_number"),
                        "source_format": "formula_derivation",
                        "derivation": "scope3_category_row_count",
                        "formula": self._first_formula_for_target(formulas, coverage_variable),
                        "source_variables": ["Scope 3 category rows in OCR table"],
                        "table_block_id": best_table.get("table_block_id"),
                    },
                }
                self.image_recognizer._merge_indicator_item(results, indicator_name, count_variable, derived_count_item)
                count_items_by_year = self._numeric_items_by_year(indicator_bucket.get(count_variable, []))

        coverage_items_by_year = self._numeric_items_by_year(indicator_bucket.get(coverage_variable, []))

        for year, count_item in count_items_by_year.items():
            if year in coverage_items_by_year:
                continue

            count_value = self._coerce_numeric_value(count_item.get("value"))
            if count_value is None:
                continue

            coverage_value = (count_value / 15.0) * 100.0
            derived_item = self._make_formula_item(
                variable_name=coverage_variable,
                value=coverage_value,
                unit="%",
                year=year,
                source_items=[count_item],
                derivation="scope3_coverage_rate",
                formula_text=self._first_formula_for_target(formulas, coverage_variable),
                source_variables=[count_variable],
            )
            self.image_recognizer._merge_indicator_item(results, indicator_name, coverage_variable, derived_item)

        if not count_items_by_year and coverage_items_by_year:
            for year, coverage_item in coverage_items_by_year.items():
                coverage_value = self._coerce_numeric_value(coverage_item.get("value"))
                if coverage_value is None:
                    continue

                coverage_unit = str(coverage_item.get("unit") or "").strip()
                coverage_ratio = coverage_value / 100.0 if "%" in coverage_unit or coverage_value > 1 else coverage_value
                count_value = coverage_ratio * 15.0
                derived_item = self._make_formula_item(
                    variable_name=count_variable,
                    value=count_value,
                    unit="category",
                    year=year,
                    source_items=[coverage_item],
                    derivation="scope3_coverage_reverse_count",
                    formula_text=self._first_formula_for_target(formulas, coverage_variable),
                    source_variables=[coverage_variable],
                )
                self.image_recognizer._merge_indicator_item(results, indicator_name, count_variable, derived_item)

    def _propagate_revenue_across_indicators(
        self,
        results: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
    ) -> None:
        indicators_with_revenue = [
            indicator_name
            for indicator_name, variable_names in indicator_variables_map.items()
            if "revenue" in variable_names
        ]
        if len(indicators_with_revenue) < 2:
            return

        canonical_revenue_by_year: Dict[int, Dict[str, Any]] = {}
        for indicator_name in indicators_with_revenue:
            items = results["all_variables_collected"].get(indicator_name, {}).get("revenue", [])
            for year, item in self._numeric_items_by_year(items).items():
                existing = canonical_revenue_by_year.get(year)
                if existing is None or self._item_confidence(item) > self._item_confidence(existing):
                    canonical_revenue_by_year[year] = item

        if not canonical_revenue_by_year:
            return

        for indicator_name in indicators_with_revenue:
            revenue_items_by_year = self._numeric_items_by_year(
                results["all_variables_collected"].get(indicator_name, {}).get("revenue", [])
            )
            for year, source_item in canonical_revenue_by_year.items():
                if year in revenue_items_by_year:
                    continue

                propagated_item = source_item.copy()
                propagated_traceability = dict(propagated_item.get("traceability") or {})
                propagated_traceability["derivation"] = "shared_indicator_revenue"
                propagated_traceability["copied_from_indicator"] = next(
                    (
                        source_indicator
                        for source_indicator in indicators_with_revenue
                        if source_item in results["all_variables_collected"].get(source_indicator, {}).get("revenue", [])
                    ),
                    None,
                )
                propagated_item["traceability"] = propagated_traceability
                propagated_item["source_type"] = "formula"
                self.image_recognizer._merge_indicator_item(results, indicator_name, "revenue", propagated_item)

    def _summarize_variable_coverage(
        self,
        results: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
    ) -> None:
        variables_with_values = 0
        variables_without_values: List[Dict[str, str]] = []

        for indicator_name, variable_names in indicator_variables_map.items():
            indicator_bucket = results["all_variables_collected"].get(indicator_name, {})
            for variable_name in variable_names:
                items = indicator_bucket.get(variable_name, [])
                if any(self.image_recognizer._has_substantive_value(item) for item in items):
                    variables_with_values += 1
                else:
                    variables_without_values.append(
                        {
                            "indicator": indicator_name,
                            "variable": variable_name,
                        }
                    )

        results["enhancement_metadata"]["variables_with_values"] = variables_with_values
        results["enhancement_metadata"]["variables_without_values"] = variables_without_values
        results["enhancement_metadata"]["variable_slot_coverage"] = {
            "total_defined_variables": sum(len(variables) for variables in indicator_variables_map.values()),
            "variables_with_values_count": variables_with_values,
            "variables_without_values_count": len(variables_without_values),
        }

    def _complete_formula_based_variables(
        self,
        results: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
        indicator_formulas_map: Dict[str, List[str]],
        all_tables: List[Dict[str, Any]],
        reporting_period_context: Dict[str, Any],
    ) -> None:
        self._propagate_revenue_across_indicators(results, indicator_variables_map)
        total_generic_formula_items = 0

        for indicator_name, variable_names in indicator_variables_map.items():
            formulas = indicator_formulas_map.get(indicator_name, [])

            if "number of Scope 3 category disclosed" in variable_names or "coverage rate of Scope 3" in variable_names:
                self._derive_scope3_category_metrics(
                    results=results,
                    indicator_name=indicator_name,
                    formulas=formulas,
                    all_tables=all_tables,
                    reporting_period_context=reporting_period_context,
                )

            if {
                "greenhouse gas emission",
                "revenue",
                "emission intensity of Scope1&2",
            }.issubset(set(variable_names)):
                self._derive_division_relationship(
                    results=results,
                    indicator_name=indicator_name,
                    formulas=formulas,
                    numerator_variable="greenhouse gas emission",
                    denominator_variable="revenue",
                    ratio_variable="emission intensity of Scope1&2",
                    ratio_scale=1.0,
                    default_denominator_unit="currency",
                    default_numerator_unit="tonnes",
                )

            if {
                "emission intensity of Scope1&2",
                "industry average emission intensity value of Scope 1&2",
                "benchmark score in Scope1&2",
            }.issubset(set(variable_names)):
                self._derive_division_relationship(
                    results=results,
                    indicator_name=indicator_name,
                    formulas=formulas,
                    numerator_variable="emission intensity of Scope1&2",
                    denominator_variable="industry average emission intensity value of Scope 1&2",
                    ratio_variable="benchmark score in Scope1&2",
                    ratio_scale=1.0,
                    default_denominator_unit="value",
                    default_numerator_unit="ratio",
                )

            if {
                "emission intensity of Scope1&2",
                "improvement rate of Scope 1&2",
            }.issubset(set(variable_names)):
                self._derive_improvement_rate(
                    results=results,
                    indicator_name=indicator_name,
                    formulas=formulas,
                    base_variable="emission intensity of Scope1&2",
                    target_variable="improvement rate of Scope 1&2",
                )

            if {
                "greenhouse gas emission of Scope 3",
                "revenue",
                "emission intensity of Scope3",
            }.issubset(set(variable_names)):
                self._derive_division_relationship(
                    results=results,
                    indicator_name=indicator_name,
                    formulas=formulas,
                    numerator_variable="greenhouse gas emission of Scope 3",
                    denominator_variable="revenue",
                    ratio_variable="emission intensity of Scope3",
                    ratio_scale=1.0,
                    default_denominator_unit="currency",
                    default_numerator_unit="tonnes",
                )

            if {
                "emission intensity of Scope3",
                "industry average emission intensity value of Scope 3",
                "benchmark score in Scope3",
            }.issubset(set(variable_names)):
                self._derive_division_relationship(
                    results=results,
                    indicator_name=indicator_name,
                    formulas=formulas,
                    numerator_variable="emission intensity of Scope3",
                    denominator_variable="industry average emission intensity value of Scope 3",
                    ratio_variable="benchmark score in Scope3",
                    ratio_scale=1.0,
                    default_denominator_unit="value",
                    default_numerator_unit="ratio",
                )

            if {
                "emission intensity of Scope3",
                "improvement rate of Scope 3",
            }.issubset(set(variable_names)):
                self._derive_improvement_rate(
                    results=results,
                    indicator_name=indicator_name,
                    formulas=formulas,
                    base_variable="emission intensity of Scope3",
                    target_variable="improvement rate of Scope 3",
                )

            total_generic_formula_items += self._apply_generic_formula_completion(
                results=results,
                indicator_name=indicator_name,
                variable_names=variable_names,
                formulas=formulas,
            )

        self._summarize_variable_coverage(results, indicator_variables_map)
        self._sort_and_trim_indicator_variables(results, indicator_variables_map)
        results["enhancement_metadata"]["generic_formula_item_count"] = total_generic_formula_items

    # Scope synonyms, most specific first. Many GHG tables label rows by emission
    # category instead of by the word "Scope": Direct = Scope 1, Indirect = Scope 2,
    # Other indirect = Scope 3. The categories are mutually independent (disjoint),
    # NOT subsets — ordering matters only because of an accidental spelling overlap
    # in the English text: "indirect" contains the letters "direct", and "other
    # indirect" contains "indirect", so a regex for the shorter phrase also matches
    # inside the longer one. Matching (and replacing) the most specific phrase first,
    # then stopping, prevents a row from being mis-tagged. (Chinese 直接/間接/其他間接
    # have no such overlap beyond 其他間接排放 trivially containing 間接排放.)
    _SCOPE_SYNONYM_PATTERNS = [
        (re.compile(r"other\s+indirect(?:\s+\w+){0,2}\s+emissions?"), "scope 3"),
        (re.compile(r"其他間接排放|其他间接排放"), "scope 3"),
        (re.compile(r"indirect(?:\s+\w+){0,2}\s+emissions?"), "scope 2"),
        (re.compile(r"間接排放|间接排放"), "scope 2"),
        (re.compile(r"direct(?:\s+\w+){0,2}\s+emissions?"), "scope 1"),
        (re.compile(r"直接排放"), "scope 1"),
    ]

    def _inject_scope_synonyms(self, normalized_text: str) -> str:
        """Rewrite a scope-synonym phrase into its canonical scope token.

        We *replace* the synonym phrase (rather than append) so the descriptive
        words are removed: e.g. "other indirect emissions" -> "scope 3". Appending
        would leave the misleading "indirect emissions" tokens in place, which then
        over-match the literal "Scope 2 (indirect emissions)" variable name on token
        overlap and steal a Scope-3 row. Replacement makes the scope token the
        dominant signal so scope-aware scoring resolves to the right variable.
        """
        if not normalized_text or "scope" in normalized_text:
            return normalized_text

        for pattern, replacement in self._SCOPE_SYNONYM_PATTERNS:
            if pattern.search(normalized_text):
                return pattern.sub(replacement, normalized_text)
        return normalized_text

    def _explicit_scope_numbers(self, normalized_text: str) -> Set[str]:
        """Scope numbers (1/2/3) explicitly named in already-normalized text.

        Run on text that has been through _inject_scope_synonyms, so synonym rows
        ("other indirect emissions" -> "scope 3") are captured here too. "scope1&2"
        / "scope 1 & 2" counts as both 1 and 2 so combined-scope variables aren't
        falsely flagged as conflicting.
        """
        nums: Set[str] = set(re.findall(r"scope\s*([123])", normalized_text))
        if re.search(r"1\s*&\s*2", normalized_text):
            nums.update({"1", "2"})
        for n in ("1", "2", "3"):
            if f"範圍{n}" in normalized_text or f"范围{n}" in normalized_text:
                nums.add(n)
        return nums

    def _score_variable_match(self, extracted_indicator: str, candidate_variable: str) -> int:
        normalized_indicator = self._inject_scope_synonyms(self._sanitize_text(extracted_indicator))
        normalized_candidate = self._inject_scope_synonyms(self._sanitize_text(candidate_variable))

        if not normalized_indicator or not normalized_candidate:
            return 0

        if normalized_indicator == normalized_candidate:
            return 100

        score = 0
        indicator_tokens = set(normalized_indicator.split())
        candidate_tokens = set(normalized_candidate.split())

        if normalized_candidate in normalized_indicator:
            score += 40
        if normalized_indicator in normalized_candidate:
            score += 35

        token_overlap = indicator_tokens & candidate_tokens
        if token_overlap:
            score += min(30, len(token_overlap) * 8)

        long_overlap = [
            token for token in token_overlap
            if len(token) >= 4 or re.search(r"[\u4e00-\u9fff]", token)
        ]
        score += min(20, len(long_overlap) * 6)

        if any(term in normalized_indicator and term in normalized_candidate for term in ["scope 1", "scope1", "範圍1", "范围1"]):
            score += 25
        if any(term in normalized_indicator and term in normalized_candidate for term in ["scope 2", "scope2", "範圍2", "范围2"]):
            score += 25
        if any(term in normalized_indicator and term in normalized_candidate for term in ["scope 3", "scope3", "範圍3", "范围3"]):
            score += 25
        if any(term in normalized_indicator and term in normalized_candidate for term in ["greenhouse gas", "ghg", "溫室氣體", "温室气体"]):
            score += 18
        if any(term in normalized_indicator and term in normalized_candidate for term in ["emission intensity", "intensity", "排放密度", "排放强度"]):
            score += 18
        if any(term in normalized_indicator and term in normalized_candidate for term in ["direct emission", "直接排放"]):
            score += 15
        if any(term in normalized_indicator and term in normalized_candidate for term in ["indirect emission", "間接排放", "间接排放"]):
            score += 15

        # Conflicting explicit scope numbers must not match on shared wording: a
        # "Scope 3 (other indirect emissions)" row outscores its own variable on
        # the "Scope 2 (indirect emissions)" name via the shared "indirect
        # emissions" tokens. Scope 1/2/3 are disjoint, so a hard clamp below the
        # match threshold (45) keeps the row from landing on the wrong scope.
        indicator_scopes = self._explicit_scope_numbers(normalized_indicator)
        candidate_scopes = self._explicit_scope_numbers(normalized_candidate)
        if indicator_scopes and candidate_scopes and indicator_scopes.isdisjoint(candidate_scopes):
            return min(score, 20)

        return score

    def _best_variable_match(
        self,
        extracted_indicator: str,
        indicator_variables_map: Dict[str, List[str]],
    ) -> Tuple[str | None, str | None, int]:
        best_indicator_name = None
        best_variable_name = None
        best_score = 0

        for indicator_name, variables in indicator_variables_map.items():
            for variable_name in variables:
                score = self._score_variable_match(extracted_indicator, variable_name)
                if score > best_score:
                    best_score = score
                    best_indicator_name = indicator_name
                    best_variable_name = variable_name

        return best_indicator_name, best_variable_name, best_score

    def _quantitative_result_template(
        self,
        pdf_name: str,
        reporting_period_context: Dict[str, Any],
        indicator_formulas_map: Dict[str, List[str]],
        indicator_variables_map: Dict[str, List[str]],
        table_page_plan: Dict[str, Any],
        keyword_page_plan: Dict[str, Any],
        variable_definition_page_plan: Dict[str, Any],
        total_pages_processed: int,
        table_block_count: int,
    ) -> Dict[str, Any]:
        results = {
            "pdf_name": pdf_name,
            "all_variables_collected": {},
            "calculation_formulas": {},
            "reporting_period_context": reporting_period_context,
            "table_page_detection": {
                "detected_pages": table_page_plan["detected_pages"],
                "page_numbers": table_page_plan["pages_to_process"],
            },
            "variable_definition_page_detection": {
                "detected_pages": variable_definition_page_plan["detected_pages"],
                "page_numbers": variable_definition_page_plan["pages_to_process"],
            },
            "enhancement_metadata": {
                "traceability_added": True,
                "processing_completed": True,
                "reporting_period_applied": reporting_period_context.get("default_year") is not None,
                "default_reporting_year": reporting_period_context.get("default_year"),
                "table_pages_detected": len(table_page_plan["pages_to_process"]),
                "keyword_pages_processed": len(keyword_page_plan["pages_to_process"]),
                "variable_definition_pages_processed": len(variable_definition_page_plan["pages_to_process"]),
                "total_pages_processed": total_pages_processed,
                "total_table_blocks_processed": table_block_count,
                "ocr_source": "datalab_json_plus_qwen_vl",
            },
        }

        for indicator_name, variable_names in indicator_variables_map.items():
            results["all_variables_collected"][indicator_name] = {
                variable_name: []
                for variable_name in variable_names
            }

        for indicator_name, formulas in indicator_formulas_map.items():
            if formulas:
                results["calculation_formulas"][indicator_name] = formulas

        return results

    def _recognize_quantitative_from_tables(
        self,
        pdf_name: str,
        indicator_variables_map: Dict[str, List[str]],
        indicator_formulas_map: Dict[str, List[str]],
        variable_definitions: Dict[str, Any],
        reporting_period_context: Dict[str, Any],
        all_tables: List[Dict[str, Any]],
        page_mode_map: Dict[int, str],
        table_page_plan: Dict[str, Any],
        keyword_page_plan: Dict[str, Any],
        variable_definition_page_plan: Dict[str, Any],
        total_pages_processed: int,
    ) -> Dict[str, Any]:
        all_variables = sorted({var for variables in indicator_variables_map.values() for var in variables})
        results = self._quantitative_result_template(
            pdf_name=pdf_name,
            reporting_period_context=reporting_period_context,
            indicator_formulas_map=indicator_formulas_map,
            indicator_variables_map=indicator_variables_map,
            table_page_plan=table_page_plan,
            keyword_page_plan=keyword_page_plan,
            variable_definition_page_plan=variable_definition_page_plan,
            total_pages_processed=total_pages_processed,
            table_block_count=len(all_tables),
        )
        skipped_tables: List[Dict[str, Any]] = []
        # Items the 7B DID extract but we then dropped (low match score / definition
        # gate). Surfaced in metadata so silent drops like the tCO₂e unit bug show up.
        dropped_items: List[Dict[str, Any]] = []

        def _record_drop(table_block: Dict[str, Any], item: Dict[str, Any], matched_variable: Optional[str], reason: str) -> None:
            record = {
                "table_block_id": table_block.get("table_block_id"),
                "page_number": table_block.get("page_number"),
                "indicator": item.get("indicator"),
                "value": item.get("value"),
                "unit": item.get("unit"),
                "matched_variable": matched_variable,
                "reason": reason,
            }
            dropped_items.append(record)
            if Config.DEBUG_EXTRACTION:
                logger.info("DROPPED_ITEM (%s): %s", reason, record)

        for table_block in all_tables:
            crop_image_path = table_block.get("crop_image_path")
            crop_image_file = None
            if crop_image_path:
                candidate = Path(crop_image_path)
                if candidate.exists():
                    crop_image_file = candidate
            # Normally a missing/absent crop means skip; in text-only mode (cached-OCR
            # re-run) there are no crops and we recognize from the table HTML alone.
            if crop_image_file is None and not self.text_only_recognition:
                continue

            page_mode = page_mode_map.get(int(table_block["page_number"]), "table_scan")
            should_analyze, diagnostics = self._should_analyze_table_with_qwen(
                table_block=table_block,
                indicator_variables_map=indicator_variables_map,
                variable_definitions=variable_definitions,
                page_mode=page_mode,
            )
            if not should_analyze:
                skipped_tables.append(
                    {
                        "table_block_id": table_block.get("table_block_id"),
                        "page_number": table_block.get("page_number"),
                        "page_mode": page_mode,
                        "reason": "low_quantitative_signal_score",
                        "diagnostics": diagnostics,
                    }
                )
                logger.info(
                    "Skipping Qwen table analysis for %s on page %s due to low quantitative signal score: %s",
                    table_block.get("table_block_id"),
                    table_block.get("page_number"),
                    diagnostics,
                )
                continue

            recognition_result = self.image_recognizer.recognize_table_json_content_unified(
                image_path=crop_image_file,
                table_block=table_block,
                all_variables=all_variables,
                reporting_period_context=reporting_period_context,
                variable_definitions=variable_definitions,
                page_mode=page_mode,
            )

            if not recognition_result or "extracted_data" not in recognition_result:
                continue

            for item in recognition_result["extracted_data"]:
                extracted_indicator = str(item.get("indicator", "")).strip()
                if not extracted_indicator:
                    continue

                matched_indicator_name, matched_variable_name, match_score = self._best_variable_match(
                    extracted_indicator,
                    indicator_variables_map,
                )

                if not matched_indicator_name or not matched_variable_name:
                    continue

                if match_score < 45:
                    _record_drop(table_block, item, matched_variable_name, f"low_match_score_{match_score}")
                    continue

                if not self._item_matches_variable_definition(
                    item=item,
                    table_block=table_block,
                    variable_name=matched_variable_name,
                    variable_definitions=variable_definitions,
                ):
                    _record_drop(table_block, item, matched_variable_name, "definition_gate")
                    continue

                item_with_page_info = item.copy()
                # Preserve the original row label before we overwrite it with the
                # canonical variable name — component aggregation needs it to tell
                # distinct component rows (Petrol vs Diesel) apart within a table.
                item_with_page_info["component_label"] = extracted_indicator
                item_with_page_info["indicator"] = matched_variable_name
                item_with_page_info["source_page"] = table_block["page_number"]
                item_with_page_info = self.image_recognizer._apply_reporting_period_to_item(
                    item_with_page_info,
                    reporting_period_context,
                )
                item_with_page_info["source_type"] = "table"
                item_with_page_info["traceability"] = {
                    "page_number": table_block["page_number"],
                    "source_format": "table_or_chart",
                    "table_block_id": table_block.get("table_block_id"),
                }

                self.image_recognizer._merge_indicator_item(
                    results,
                    matched_indicator_name,
                    matched_variable_name,
                    item_with_page_info,
                )

        # Sum split component rows (Petrol + Diesel under one Scope) before the
        # Scope1+2 total is derived, so the total builds on the aggregated scopes.
        self._aggregate_table_components(results, indicator_variables_map)

        for indicator_name, variable_names in indicator_variables_map.items():
            for variable_name in variable_names:
                self._derive_total_scope_1_2_if_missing(results, indicator_name, variable_name)

        self._complete_formula_based_variables(
            results=results,
            indicator_variables_map=indicator_variables_map,
            indicator_formulas_map=indicator_formulas_map,
            all_tables=all_tables,
            reporting_period_context=reporting_period_context,
        )
        results["enhancement_metadata"]["skipped_qwen_tables"] = skipped_tables
        results["enhancement_metadata"]["skipped_qwen_table_count"] = len(skipped_tables)
        results["enhancement_metadata"]["dropped_extracted_items"] = dropped_items
        results["enhancement_metadata"]["dropped_extracted_item_count"] = len(dropped_items)

        save_json(results, self._get_pdf_quantitative_result_file(pdf_name))
        return results

    # The three variables that hold absolute per-scope GHG emissions.
    _SCOPE_ABS_VARS = (
        "Scope 1 (direct emissions)",
        "Scope 2 (indirect emissions)",
        "greenhouse gas emission of Scope 3",
    )

    def _has_any_scope_value(self, results: Dict[str, Any]) -> bool:
        for vmap in results.get("all_variables_collected", {}).values():
            if not isinstance(vmap, dict):
                continue
            for vname in self._SCOPE_ABS_VARS:
                if vmap.get(vname):
                    return True
        return False

    def _vision_fallback_for_scopes(
        self,
        pdf_name: str,
        pdf_path: Optional[Path],
        results: Dict[str, Any],
        indicator_variables_map: Dict[str, List[str]],
        variable_definitions: Dict[str, Any],
        reporting_period_context: Dict[str, Any],
        candidate_pages: List[int],
    ) -> None:
        """Last-resort recovery: when table extraction found NO scope emissions at all,
        read the candidate pages as full PAGE IMAGES (some reports put the numbers in an
        infographic/callout that Chandra never turns into a <table>; e.g. 0693 p19's
        "CO2 emissions 130 thousand t-CO2"). Purely additive — only runs when scopes are
        empty, and every item still goes through matching + the definition gate.
        """
        if not Config.VISION_SCOPE_FALLBACK or self._has_any_scope_value(results):
            return
        if not pdf_path or not Path(pdf_path).exists():
            logger.warning("Vision fallback skipped for %s: no PDF at %s", pdf_name, pdf_path)
            return

        # Scan the whole candidate set — it's already the filtered set of pages with
        # tables or GHG keywords, so it's small; an arbitrary cap could drop the very
        # page that holds the figure.
        pages = sorted({int(p) for p in candidate_pages})
        if not pages:
            return
        logger.info("Vision fallback for %s: no scope values from tables, scanning %d page images", pdf_name, len(pages))

        all_variables = sorted({v for vs in indicator_variables_map.values() for v in vs})
        self.image_recognizer._ensure_pdf_images(Path(pdf_path))
        img_dir = self.image_recognizer._get_pdf_image_dir(pdf_name)

        added = 0
        for page in pages:
            img = img_dir / ("page_%03d.png" % page)
            if not img.exists():
                continue
            recognition = self.image_recognizer.recognize_page_ghg_vision(
                image_path=img,
                all_variables=all_variables,
                reporting_period_context=reporting_period_context,
            )
            page_items = (recognition or {}).get("extracted_data", [])
            # Page context for the LLM match-fallback: the other indicators 7B read
            # on this page (helps disambiguate scope, e.g. "no Scope 1, only
            # purchased electricity" → a lone power CO2 figure is Scope 2).
            page_context = "; ".join(
                str(it.get("indicator", "")).strip()
                for it in page_items
                if str(it.get("indicator", "")).strip()
            )
            for item in page_items:
                extracted_indicator = str(item.get("indicator", "")).strip()
                if not extracted_indicator:
                    continue
                # Hard guard against zero/placeholder values: the page-GHG prompt is
                # told to omit absent/zero figures, but enforce it here too so a
                # stray 0/null can never be admitted and pollute trim/aggregation
                # (0693 was returning a placeholder "greenhouse gas emission"=0 that
                # displaced the real 130). A genuine 0-emission scope is vanishingly
                # rare in these reports and not worth the false positives.
                raw_value = item.get("value")
                try:
                    if raw_value is None or float(str(raw_value).replace(",", "")) <= 0:
                        continue
                except (TypeError, ValueError):
                    continue
                matched_indicator_name, matched_variable_name, match_score = self._best_variable_match(
                    extracted_indicator, indicator_variables_map
                )
                # Deterministic matcher is the primary path. When it can't route a
                # vision item (business label unlike any variable name, e.g.
                # "CO2 emissions" / "Purchased Electricity"), fall back to the LLM
                # semantic classifier instead of dropping the figure.
                if not matched_variable_name or match_score < 45:
                    all_variables = sorted(
                        {v for vs in indicator_variables_map.values() for v in vs}
                    )
                    llm_variable = self.image_recognizer.classify_indicator_to_variable(
                        extracted_indicator=extracted_indicator,
                        value=item.get("value"),
                        unit=item.get("unit"),
                        candidate_variables=all_variables,
                        variable_definitions=variable_definitions,
                        page_context=page_context,
                    )
                    if not llm_variable:
                        continue
                    matched_variable_name = llm_variable
                    matched_indicator_name = next(
                        (ind for ind, vs in indicator_variables_map.items() if llm_variable in vs),
                        matched_indicator_name,
                    )
                if not self._item_matches_variable_definition(
                    item=item,
                    table_block={},
                    variable_name=matched_variable_name,
                    variable_definitions=variable_definitions,
                    from_page_image=True,
                ):
                    continue
                vision_item = item.copy()
                vision_item["component_label"] = extracted_indicator
                vision_item["indicator"] = matched_variable_name
                vision_item["source_page"] = page
                # The page-GHG prompt doesn't ask for a confidence; give downstream
                # trim/ranking a sensible default (same as the derived Scope1+2 total).
                if vision_item.get("confidence") is None:
                    vision_item["confidence"] = 0.8
                vision_item = self.image_recognizer._apply_reporting_period_to_item(
                    vision_item, reporting_period_context
                )
                vision_item["source_type"] = "page_image"
                vision_item["traceability"] = {
                    "page_number": page,
                    "source_format": "page_image_vision_fallback",
                }
                self.image_recognizer._merge_indicator_item(
                    results, matched_indicator_name, matched_variable_name, vision_item
                )
                added += 1

        if added:
            self._aggregate_table_components(results, indicator_variables_map)
            for indicator_name, variable_names in indicator_variables_map.items():
                for variable_name in variable_names:
                    self._derive_total_scope_1_2_if_missing(results, indicator_name, variable_name)
            self._sort_and_trim_indicator_variables(results, indicator_variables_map)
            results["enhancement_metadata"]["vision_fallback_items_added"] = added
            save_json(results, self._get_pdf_quantitative_result_file(pdf_name))
        logger.info("Vision fallback for %s: added %d items from page images", pdf_name, added)

    def _cleanup_temp_artifacts(self, pdf_name: str) -> None:
        table_crops_dir = self._get_temp_table_crops_dir(pdf_name)
        if table_crops_dir.exists():
            shutil.rmtree(table_crops_dir, ignore_errors=True)

        for legacy_dir_name in ("raw_pages", "tables", "table_crops"):
            legacy_dir = self._get_pdf_output_dir(pdf_name) / legacy_dir_name
            if legacy_dir.exists():
                shutil.rmtree(legacy_dir, ignore_errors=True)

        if Config.CLEANUP_IMAGES_AFTER_RECOGNITION:
            self.image_recognizer._cleanup_pdf_images(pdf_name)

    def process_single_pdf_from_cache(
        self,
        keywords_config_path: Path,
        variable_definitions_path: Path,
        pdf_name: str,
    ) -> Dict[str, Any]:
        """Re-run only the Qwen/matching/aggregation downstream from cached OCR.

        Skips the slow Chandra OCR by reusing ``<pdf>_ocr_output.json`` (table HTML,
        text, bbox, reporting period, page plans are all cached there). Recognition
        runs text-only (no table crop image, which is cleaned up after each full run)
        off the cached table HTML — fast iteration on extraction/matching logic.
        """
        ocr_file = self._get_pdf_ocr_output_file(pdf_name)
        if not ocr_file.exists():
            logger.error("No cached OCR output for %s at %s; run a full OCR pass first", pdf_name, ocr_file)
            return {"pdf_name": pdf_name, "status": "failed", "error": "no_cached_ocr_output"}

        logger.info("Reusing cached OCR for %s (text-only downstream re-run)", pdf_name)
        with open(ocr_file, "r", encoding="utf-8") as fh:
            ocr_output = json.load(fh)

        keywords_config = self._load_keywords_config(keywords_config_path)
        variable_definitions = self._load_variable_definitions(variable_definitions_path)
        indicator_variables_map, indicator_formulas_map = self._extract_quantitative_indicator_maps(keywords_config)

        reporting_period_context = ocr_output.get("reporting_period_context", {}) or {}

        runtime_tables: List[Dict[str, Any]] = []
        for table_record in ocr_output.get("tables", []):
            if not isinstance(table_record, dict):
                continue
            runtime_block = dict(table_record)
            runtime_block["crop_image_path"] = None
            runtime_tables.append(runtime_block)

        page_mode_map: Dict[int, str] = {}
        for page_record in ocr_output.get("pages", []):
            if not isinstance(page_record, dict):
                continue
            try:
                page_mode_map[int(page_record["page_number"])] = page_record.get("page_mode", "table_scan")
            except (KeyError, TypeError, ValueError):
                continue

        table_detection = ocr_output.get("table_page_detection", {}) or {}
        table_page_plan = {
            "detected_pages": table_detection.get("detected_pages", []),
            "pages_to_process": table_detection.get("page_numbers", []),
        }
        keyword_page_plan = ocr_output.get("keyword_page_plan") or {"detected_pages": [], "pages_to_process": []}
        keyword_page_plan.setdefault("pages_to_process", [])
        definition_detection = ocr_output.get("variable_definition_page_detection", {}) or {}
        variable_definition_page_plan = {
            "detected_pages": definition_detection.get("detected_pages", []),
            "pages_to_process": definition_detection.get("page_numbers", []),
        }

        self.text_only_recognition = True
        try:
            quantitative_results = self._recognize_quantitative_from_tables(
                pdf_name=pdf_name,
                indicator_variables_map=indicator_variables_map,
                indicator_formulas_map=indicator_formulas_map,
                variable_definitions=variable_definitions,
                reporting_period_context=reporting_period_context,
                all_tables=runtime_tables,
                page_mode_map=page_mode_map,
                table_page_plan=table_page_plan,
                keyword_page_plan=keyword_page_plan,
                variable_definition_page_plan=variable_definition_page_plan,
                total_pages_processed=len(ocr_output.get("pages", [])),
            )
        finally:
            self.text_only_recognition = False

        pdf_path = Config.PDF_INPUT_DIR / ("%s.pdf" % pdf_name)
        if not pdf_path.exists():
            pdf_path = Path("pdf_input") / ("%s.pdf" % pdf_name)
        candidate_pages = list(table_page_plan["pages_to_process"]) + list(keyword_page_plan.get("pages_to_process", []))
        self._vision_fallback_for_scopes(
            pdf_name=pdf_name,
            pdf_path=pdf_path,
            results=quantitative_results,
            indicator_variables_map=indicator_variables_map,
            variable_definitions=variable_definitions,
            reporting_period_context=reporting_period_context,
            candidate_pages=candidate_pages,
        )

        return {
            "pdf_name": pdf_name,
            "reused_cached_ocr": True,
            "quantitative_result_file": str(self._get_pdf_quantitative_result_file(pdf_name)),
            "table_count": len(runtime_tables),
            "indicator_count": len(quantitative_results.get("all_variables_collected", {})),
        }

    def process_single_pdf(
        self,
        keywords_config_path: Path,
        variable_definitions_path: Path,
        keyword_results_dir: Path,
        pdf_path: Path,
    ) -> Dict[str, Any]:
        pdf_name = pdf_path.stem
        logger.info(f"Starting Datalab OCR test for {pdf_name}")

        keywords_config = self._load_keywords_config(keywords_config_path)
        variable_definitions = self._load_variable_definitions(variable_definitions_path)
        indicator_variables_map, indicator_formulas_map = self._extract_quantitative_indicator_maps(keywords_config)
        keyword_results = self._load_keyword_results(keyword_results_dir, pdf_name)

        self.image_recognizer._ensure_pdf_images(pdf_path)
        text_content = self.image_recognizer._load_pdf_text_content(pdf_name, pdf_path=pdf_path)
        reporting_period_context = self.image_recognizer.extract_reporting_period_context(pdf_name, pdf_path=pdf_path)

        keyword_page_plan = self.image_recognizer._build_page_processing_plan(keyword_results)
        table_page_plan = self.image_recognizer._build_report_table_page_plan(
            text_content,
            list(indicator_variables_map.keys()),
        )
        variable_definition_page_plan = self._build_variable_definition_page_plan(
            text_content=text_content,
            indicator_variables_map=indicator_variables_map,
            variable_definitions=variable_definitions,
        )

        pages_to_process: List[int] = []
        page_mode_map: Dict[int, str] = {}

        for page_num in table_page_plan["pages_to_process"]:
            pages_to_process.append(page_num)
            page_mode_map[page_num] = "table_scan"

        for page_num in keyword_page_plan["pages_to_process"]:
            if page_num not in page_mode_map:
                pages_to_process.append(page_num)
                page_mode_map[page_num] = "keyword_page"

        for page_num in variable_definition_page_plan["pages_to_process"]:
            if page_num not in page_mode_map:
                pages_to_process.append(page_num)
                page_mode_map[page_num] = "variable_definition_page"

        output_dir = self._get_pdf_output_dir(pdf_name)
        ensure_directory_exists(output_dir)
        ensure_directory_exists(self._get_temp_table_crops_dir(pdf_name))

        ocr_tables: List[Dict[str, Any]] = []
        runtime_tables: List[Dict[str, Any]] = []
        page_ocr_records: List[Dict[str, Any]] = []
        pdf_image_dir = self.image_recognizer._get_pdf_image_dir(pdf_name)

        try:
            for page_num in pages_to_process:
                image_file = pdf_image_dir / f"page_{page_num:03d}.png"
                if not image_file.exists():
                    logger.warning(f"Image file missing for {pdf_name} page {page_num}: {image_file}")
                    continue

                page_result = self._run_page_ocr(image_file)
                page_result["page_number"] = page_num
                page_result["page_mode"] = page_mode_map.get(page_num, "keyword_page")

                table_blocks = self._extract_table_blocks(
                    page_result=page_result,
                    pdf_name=pdf_name,
                    page_num=page_num,
                    image_file=image_file,
                )

                page_ocr_records.append(
                    {
                        "page_number": page_num,
                        "page_mode": page_result["page_mode"],
                        "parse_quality_score": page_result.get("parse_quality_score"),
                        "runtime": page_result.get("runtime"),
                        "table_count": len(table_blocks),
                    }
                )

                ocr_tables.extend(self._table_record_for_ocr_json(table_block) for table_block in table_blocks)
                runtime_tables.extend(self._table_runtime_record(table_block) for table_block in table_blocks)

            ocr_output = {
                "pdf_name": pdf_name,
                "ocr_backend": self.ocr_backend,
                "requested_mode": self.convert_mode,
                "source_format_version": self.RESULT_FORMAT_VERSION,
                "reporting_period_context": reporting_period_context,
                "variable_definitions_version": "quantitative_variable_definitions_v1",
                "table_page_detection": {
                    "detected_pages": table_page_plan["detected_pages"],
                    "page_numbers": table_page_plan["pages_to_process"],
                },
                "variable_definition_page_detection": {
                    "detected_pages": variable_definition_page_plan["detected_pages"],
                    "page_numbers": variable_definition_page_plan["pages_to_process"],
                },
                "keyword_page_plan": keyword_page_plan,
                "pages": page_ocr_records,
                "tables": ocr_tables,
            }
            save_json(ocr_output, self._get_pdf_ocr_output_file(pdf_name))

            quantitative_results = self._recognize_quantitative_from_tables(
                pdf_name=pdf_name,
                indicator_variables_map=indicator_variables_map,
                indicator_formulas_map=indicator_formulas_map,
                variable_definitions=variable_definitions,
                reporting_period_context=reporting_period_context,
                all_tables=runtime_tables,
                page_mode_map=page_mode_map,
                table_page_plan=table_page_plan,
                keyword_page_plan=keyword_page_plan,
                variable_definition_page_plan=variable_definition_page_plan,
                total_pages_processed=len(page_ocr_records),
            )

            self._vision_fallback_for_scopes(
                pdf_name=pdf_name,
                pdf_path=pdf_path,
                results=quantitative_results,
                indicator_variables_map=indicator_variables_map,
                variable_definitions=variable_definitions,
                reporting_period_context=reporting_period_context,
                candidate_pages=list(table_page_plan["pages_to_process"]) + list(keyword_page_plan.get("pages_to_process", [])),
            )

            return {
                "pdf_name": pdf_name,
                "ocr_output_file": str(self._get_pdf_ocr_output_file(pdf_name)),
                "quantitative_result_file": str(self._get_pdf_quantitative_result_file(pdf_name)),
                "table_count": len(ocr_tables),
                "indicator_count": len(quantitative_results.get("all_variables_collected", {})),
            }
        finally:
            self._cleanup_temp_artifacts(pdf_name)

    def process_specific_pdfs(
        self,
        keywords_config_path: Path,
        variable_definitions_path: Path,
        keyword_results_dir: Path,
        pdf_paths: List[Path],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        deduped_pdf_paths = self.image_recognizer._dedupe_pdf_paths(pdf_paths)

        for pdf_path in deduped_pdf_paths:
            try:
                if self.reuse_cached_ocr:
                    results[pdf_path.stem] = self.process_single_pdf_from_cache(
                        keywords_config_path=keywords_config_path,
                        variable_definitions_path=variable_definitions_path,
                        pdf_name=pdf_path.stem,
                    )
                    continue
                results[pdf_path.stem] = self.process_single_pdf(
                    keywords_config_path=keywords_config_path,
                    variable_definitions_path=variable_definitions_path,
                    keyword_results_dir=keyword_results_dir,
                    pdf_path=pdf_path,
                )
            except Exception as exc:
                logger.error(f"Datalab OCR processing failed for {pdf_path.name}: {exc}", exc_info=True)
                results[pdf_path.stem] = {
                    "pdf_name": pdf_path.stem,
                    "status": "failed",
                    "error": str(exc),
                }

        return results
