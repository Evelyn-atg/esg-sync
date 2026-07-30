"""
Numeric Extractor: extracts all text blocks containing numeric values from a PDF.

Sources:
1. Text paragraphs (from paragraphs/ directory) — filtered for numeric content
2. Table pages (detected via heuristic scoring) — OCR'd to get structured table content

Output: numeric_extracts/<pdf_name>/numeric_blocks.json
"""

import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import Config
from src.utils import setup_logger, ensure_directory_exists, save_json, GPUFatalError

logger = setup_logger(__name__, "logs/numeric_extractor.log")

# Patterns that indicate a line is just a page number, TOC entry, or header/footer
_PAGE_NUMBER_PATTERN = re.compile(r'^\s*\d{1,3}\s*$')
_TOC_PATTERN = re.compile(
    r'^.*?[\.·…]{3,}\s*\d{1,3}\s*$'  # "Section name ..... 42"
    r'|^(?:第\s*)?[一二三四五六七八九十\d]+[\.、]\s*.{2,30}\s+\d{1,3}$'  # "一、概述 5"
)
_HEADER_FOOTER_PATTERN = re.compile(
    r'^\s*(?:page|第\s*\d+\s*页|頁)\s*\d*\s*$',
    re.IGNORECASE,
)

# Numeric value pattern: at least one meaningful number (not just a year alone)
_NUMERIC_VALUE_PATTERN = re.compile(
    r'(?<![A-Za-z])\d[\d,]*(?:\.\d+)?'
)

# Year-only pattern: lines that are just a year reference
_YEAR_ONLY_PATTERN = re.compile(
    r'^[\s\d/\-年月日財财FYfy]*$'
)

# Chandra HF backend does true batched inference (one vision-encoder forward +
# one batched LLM decode per batch).
# Default 8 is safe for single-process H100 80GB; tune up to 16-32 if
# nvidia-smi shows headroom, down if you hit OOM.
# For multi-stream mode (multiple Python processes sharing one GPU), set
# OCR_BATCH_SIZE env var to a smaller value (e.g. 2 for 5 streams).
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "8"))


def _current_gpu_id() -> str:
    """Best-effort GPU id for log tagging. Reads CUDA_VISIBLE_DEVICES first
    (works even before torch is imported), falls back to torch.cuda.current_device().
    Returns a short string like ``"0"``, ``"0,1"``, or ``"cpu"``.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return cvd.strip()
    try:
        import torch
        if torch.cuda.is_available():
            return str(torch.cuda.current_device())
    except Exception:
        pass
    return "cpu"


def _physical_gpu_id() -> Optional[str]:
    """Single physical GPU id for nvidia-smi ``--id``, or ``None`` if ambiguous
    (no CUDA_VISIBLE_DEVICES, multi-GPU visible, etc).
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd or "," in cvd:
        return None
    return cvd


class _GpuUtilSampler:
    """Background nvidia-smi sampler. While ``start()``'d, polls
    ``utilization.gpu`` every ``interval_s`` seconds and keeps the samples.
    ``stop()`` + ``avg`` gives the mean GPU util % during the window.

    Silent no-op if nvidia-smi is missing, GPU id is ambiguous, or any call
    fails — the main OCR path never breaks because of telemetry.
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: List[int] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_id = _physical_gpu_id()

    def start(self) -> "_GpuUtilSampler":
        if self._gpu_id is None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self._gpu_id}",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=1.0,
                    stderr=subprocess.DEVNULL,
                )
                self.samples.append(int(out.decode().strip()))
            except Exception:
                # nvidia-smi flake / no GPU / permission — just skip the tick
                pass
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def avg(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def max(self) -> int:
        return max(self.samples) if self.samples else 0

    @property
    def min(self) -> int:
        return min(self.samples) if self.samples else 0

class NumericExtractor:
    """Extracts numeric text blocks from PDF text and OCR table pages."""

    def __init__(self):
        self.output_dir = Config.NUMERIC_EXTRACT_DIR
        self.paragraph_dir = Config.PARAGRAPH_OUTPUT_DIR
        self.text_output_dir = Config.TEXT_OUTPUT_DIR

    def _sanitize_pdf_name(self, pdf_name: str) -> str:
        return "".join(c for c in pdf_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

    def _get_output_path(self, pdf_name: str) -> Path:
        return self.output_dir / pdf_name / "numeric_blocks.json"

    def _has_existing_result(self, pdf_name: str) -> bool:
        output_path = self._get_output_path(pdf_name)
        if not output_path.exists():
            return False
        if output_path.stat().st_size == 0:
            return False
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            return False

    def _is_noise_line(self, text: str) -> bool:
        """Check if a text block is just page number, TOC, or header/footer noise."""
        stripped = text.strip()
        if not stripped:
            return True
        if _PAGE_NUMBER_PATTERN.match(stripped):
            return True
        if _TOC_PATTERN.match(stripped):
            return True
        if _HEADER_FOOTER_PATTERN.match(stripped):
            return True
        return False

    def _has_meaningful_numbers(self, text: str) -> bool:
        """Check if text contains meaningful numeric values (not just years or page refs)."""
        numbers = _NUMERIC_VALUE_PATTERN.findall(text)
        if not numbers:
            return False

        # Filter out numbers that are just years (2019-2026) or very small (page numbers <=3 digits alone)
        meaningful_count = 0
        for num_str in numbers:
            # Remove commas for checking
            clean_num = num_str.replace(',', '')
            try:
                val = float(clean_num)
            except ValueError:
                continue

            # Skip if it looks like a standalone year
            if 1900 <= val <= 2099 and '.' not in num_str and ',' not in num_str:
                continue

            meaningful_count += 1

        return meaningful_count > 0

    def _extract_text_blocks(self, pdf_name: str) -> List[Dict[str, Any]]:
        """Extract numeric text blocks from the paragraphs file."""
        sanitized = self._sanitize_pdf_name(pdf_name)
        paragraph_file = self.paragraph_dir / sanitized / "paragraphs.json"

        if not paragraph_file.exists():
            logger.warning(f"No paragraphs file found for {pdf_name}: {paragraph_file}")
            return []

        with open(paragraph_file, 'r', encoding='utf-8') as f:
            paragraphs = json.load(f)

        numeric_blocks = []
        for para in paragraphs:
            text = para.get('text', '')

            if self._is_noise_line(text):
                continue

            if not self._has_meaningful_numbers(text):
                continue

            numeric_blocks.append({
                "source": "text",
                "page": para.get('page_number', 0),
                "content": text,
                "has_table": False,
                "paragraph_id": para.get('id', ''),
            })

        logger.info(f"[{pdf_name}] Extracted {len(numeric_blocks)} numeric text blocks from {len(paragraphs)} paragraphs")
        return numeric_blocks

    def _extract_table_blocks(self, pdf_name: str, pdf_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Detect table pages via heuristic scoring, then OCR them with Chandra to get
        structured table content (HTML). Falls back to raw text if OCR unavailable.
        """
        from src.image_recognizer import ImageRecognizer
        from src.pdf_extractor import PDFExtractor

        recognizer = ImageRecognizer()

        # Load PDF text content for table page detection
        text_content = recognizer._load_pdf_text_content(pdf_name, pdf_path=pdf_path)
        if not text_content:
            logger.warning(f"[{pdf_name}] No text content available for table detection")
            return []

        # Detect table pages using existing heuristic
        detected_pages = recognizer._detect_report_table_pages(text_content)
        if not detected_pages:
            logger.info(f"[{pdf_name}] No table pages detected")
            return []

        page_numbers = [item["page_number"] for item in detected_pages]
        logger.info(f"[{pdf_name}] Detected {len(page_numbers)} table pages: {page_numbers}")

        # Try OCR with Chandra if available
        ocr_results = self._ocr_table_pages(pdf_name, pdf_path, page_numbers)

        table_blocks = []
        for page_info in detected_pages:
            page_num = page_info["page_number"]

            # Use OCR result if available, else fall back to PyMuPDF text
            if page_num in ocr_results and ocr_results[page_num]:
                content = ocr_results[page_num]
                source = "table_ocr"
            else:
                page_content_map = recognizer._parse_page_content_map(text_content)
                content = page_content_map.get(page_num, "")
                source = "table_page"

            if not content:
                continue

            table_blocks.append({
                "source": source,
                "page": page_num,
                "content": content,
                "has_table": True,
                "detection_score": page_info.get("score", 0),
                "detection_signals": page_info.get("signals", []),
            })

        ocr_count = sum(1 for b in table_blocks if b["source"] == "table_ocr")
        text_count = sum(1 for b in table_blocks if b["source"] == "table_page")
        logger.info(f"[{pdf_name}] Extracted {len(table_blocks)} table blocks (OCR={ocr_count}, text_fallback={text_count})")
        return table_blocks

    def _ocr_table_pages(
        self,
        pdf_name: str,
        pdf_path: Optional[Path],
        page_numbers: List[int],
    ) -> Dict[int, str]:
        """
        OCR specific pages using Chandra. Returns {page_number: html_content}.
        Uses cached results for pages already OCR'd; runs Chandra only on missing pages.
        If Chandra is not available or OCR fails, returns empty dict for those pages.
        """
        import os
        from src.image_recognizer import ImageRecognizer

        ocr_results: Dict[int, str] = {}
        pages_to_ocr: List[int] = list(page_numbers)

        # Load whatever we already have from cache
        cached_ocr_file = Config.CHANDRA_OCR_RESULT_DIR / pdf_name / f"{pdf_name}_ocr_output.json"
        if cached_ocr_file.exists():
            cached = self._load_cached_ocr(cached_ocr_file, page_numbers)
            ocr_results.update(cached)
            pages_to_ocr = [p for p in page_numbers if p not in cached]

        if not pages_to_ocr:
            logger.info(f"[{pdf_name}] All {len(page_numbers)} table pages served from OCR cache")
            return ocr_results

        logger.info(
            f"[{pdf_name}] OCR cache hit {len(ocr_results)} pages, "
            f"need to OCR {len(pages_to_ocr)} more: {pages_to_ocr}"
        )

        # Need the PDF file to render page images for OCR
        if pdf_path is None or not pdf_path.exists():
            pdf_path = Config.PDF_INPUT_DIR / f"{pdf_name}.pdf"
            if not pdf_path.exists():
                logger.warning(f"[{pdf_name}] PDF not found for OCR — cannot OCR missing pages")
                return ocr_results

        # Render page images
        recognizer = ImageRecognizer()
        pdf_image_dir = recognizer._ensure_pdf_images(pdf_path)

        # Force offline mode so Chandra/transformers doesn't try to reach HuggingFace
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # Create OCR tester ONCE and reuse for all pages
        try:
            from src.chandra_ocr_tester import ChandraOCRTester
            ocr_tester = ChandraOCRTester()
            # Pre-load the model so it's ready for all pages
            ocr_tester._ensure_chandra_manager()
        except (ImportError, Exception) as e:
            logger.warning(f"[{pdf_name}] Chandra OCR not available: {e}. Cannot OCR missing pages")
            return ocr_results

        new_ocr: Dict[int, str] = {}
        new_tables: List[Dict[str, Any]] = []   # rich per-table records (block id, bbox, quality)
        new_page_meta: List[Dict[str, Any]] = []  # rich per-page records (mode, quality, table_count)

        # Pre-filter: only pages whose image file exists
        valid_pages: List[int] = []
        valid_images: List[Path] = []
        for page_num in pages_to_ocr:
            image_file = pdf_image_dir / f"page_{page_num:03d}.png"
            if not image_file.exists():
                logger.warning(f"[{pdf_name}] Page image not found: {image_file}")
                continue
            valid_pages.append(page_num)
            valid_images.append(image_file)

        if not valid_pages:
            logger.info(f"[{pdf_name}] No OCR-able page images found")
        else:
            # Process in batches — HF backend does a single batched forward
            # per batch (~3-5x throughput vs per-image calls).
            batch_size = OCR_BATCH_SIZE
            total_batches = (len(valid_pages) + batch_size - 1) // batch_size
            gpu_id = _current_gpu_id()
            ocr_start_t = time.time()

            # Per-PDF cumulative counters for the summary line
            pages_done = 0
            tokens_done = 0
            sum_img_load_ms = 0.0
            sum_gen_wall_ms = 0.0
            sum_gen_gpu_ms = 0.0
            sum_postproc_ms = 0.0
            gpu_util_samples_sum = 0.0
            gpu_util_samples_n = 0

            logger.info(
                f"[esg-ocr][gpu={gpu_id}][{pdf_name}] OCR batching: "
                f"{len(valid_pages)} pages → {total_batches} batch(es) of ≤{batch_size}"
            )

            for batch_start in range(0, len(valid_pages), batch_size):
                batch_pages = valid_pages[batch_start:batch_start + batch_size]
                batch_images = valid_images[batch_start:batch_start + batch_size]
                batch_idx = batch_start // batch_size + 1

                # --- batch call + GPU util sampling + timing breakdown ---
                timing: Dict[str, float] = {}
                sampler = _GpuUtilSampler().start()
                try:
                    batch_results = ocr_tester._run_page_ocr_batch(batch_images, timing=timing)
                except GPUFatalError as e:
                    # GPU fatal error - stop sampler, log critical error, and exit immediately
                    sampler.stop()
                    logger.critical(
                        f"[esg-ocr][gpu={gpu_id}][{pdf_name}] FATAL GPU ERROR in batch {batch_idx}/{total_batches}: {e}"
                    )
                    logger.critical("Worker exiting immediately. GPU marked as unhealthy.")
                    # Re-raise to trigger immediate worker exit
                    raise
                except Exception as e:
                    sampler.stop()
                    logger.warning(
                        f"[esg-ocr][gpu={gpu_id}][{pdf_name}] Batch {batch_idx}/{total_batches} "
                        f"failed entirely (pages {batch_pages}): {e}"
                    )
                    continue
                sampler.stop()

                img_load_ms = timing.get("img_load_ms", 0.0)
                gen_wall_ms = timing.get("generate_wall_ms", 0.0)
                gen_gpu_ms = timing.get("generate_gpu_ms", gen_wall_ms)

                # --- postprocess each page ---
                postproc_t0 = time.perf_counter()
                batch_tokens = 0
                for page_num, result in zip(batch_pages, batch_results):
                    try:
                        if result.get("status") == "failed":
                            logger.warning(
                                f"[{pdf_name}] OCR failed for page {page_num}: {result.get('error')}"
                            )
                            continue

                        # Prefer HTML (structured tables), fall back to markdown
                        html_content = result.get("html", "")
                        if html_content:
                            new_ocr[page_num] = html_content
                        elif result.get("markdown"):
                            new_ocr[page_num] = result["markdown"]

                        # Rich format: split page into per-table blocks (bbox, block id, section)
                        image_file = pdf_image_dir / f"page_{page_num:03d}.png"
                        table_blocks = ocr_tester._extract_table_blocks(
                            page_result=result, pdf_name=pdf_name,
                            page_num=page_num, image_file=image_file,
                        )
                        new_tables.extend(
                            ocr_tester._table_record_for_ocr_json(tb) for tb in table_blocks
                        )
                        new_page_meta.append({
                            "page_number": str(page_num),
                            "page_mode": "table_scan",
                            "parse_quality_score": result.get("parse_quality_score"),
                            "runtime": result.get("runtime"),
                            "table_count": len(table_blocks),
                        })
                        pages_done += 1
                        batch_tokens += int(result.get("token_count") or 0)
                    except Exception as e:
                        logger.warning(f"[{pdf_name}] Post-OCR processing failed for page {page_num}: {e}")
                        continue
                postproc_ms = (time.perf_counter() - postproc_t0) * 1000
                tokens_done += batch_tokens

                # --- accumulate ---
                sum_img_load_ms += img_load_ms
                sum_gen_wall_ms += gen_wall_ms
                sum_gen_gpu_ms += gen_gpu_ms
                sum_postproc_ms += postproc_ms
                if sampler.samples:
                    gpu_util_samples_sum += sum(sampler.samples)
                    gpu_util_samples_n += len(sampler.samples)

                # --- per-batch log ---
                batch_wall_s = (img_load_ms + gen_wall_ms + postproc_ms) / 1000
                batch_pps = len(batch_pages) / batch_wall_s if batch_wall_s > 0 else 0.0
                # CPU overhead = (generate wall) - (generate GPU). Includes
                # tokenize / pad / batch_decode inside Chandra.
                cpu_overhead_ms = gen_wall_ms - gen_gpu_ms
                gpu_pct = (gen_gpu_ms / gen_wall_ms * 100) if gen_wall_ms > 0 else 0.0
                tokens_per_s = batch_tokens / (gen_gpu_ms / 1000) if gen_gpu_ms > 0 else 0.0
                overall_elapsed = time.time() - ocr_start_t
                overall_pps = pages_done / overall_elapsed if overall_elapsed > 0 else 0.0

                util_str = (
                    f"util≈{sampler.avg:.0f}%"
                    if sampler.samples
                    else "util=n/a"
                )
                logger.info(
                    f"[esg-ocr][gpu={gpu_id}][{pdf_name}] "
                    f"Batch {batch_idx}/{total_batches} "
                    f"(pages {batch_pages[0]}-{batch_pages[-1]}, "
                    f"{batch_tokens} tok): "
                    f"wall {batch_wall_s:.2f}s = "
                    f"img_load {img_load_ms/1000:.2f}s + "
                    f"generate {gen_wall_ms/1000:.2f}s "
                    f"(gpu {gen_gpu_ms/1000:.2f}s, "
                    f"cpu_overhead {cpu_overhead_ms/1000:.2f}s, "
                    f"gpu_inside {gpu_pct:.0f}%) + "
                    f"postproc {postproc_ms/1000:.2f}s | "
                    f"{batch_pps:.2f} pages/s, "
                    f"{tokens_per_s:.0f} tok/s, "
                    f"{util_str} | "
                    f"cumulative {pages_done} pages in {overall_elapsed:.2f}s, "
                    f"avg {overall_pps:.2f} pages/s"
                )

            # --- final summary for this PDF ---
            ocr_total_t = time.time() - ocr_start_t
            total_wall_ms = sum_img_load_ms + sum_gen_wall_ms + sum_postproc_ms
            total_cpu_overhead_ms = sum_gen_wall_ms - sum_gen_gpu_ms
            overall_gpu_pct = (sum_gen_gpu_ms / sum_gen_wall_ms * 100) if sum_gen_wall_ms > 0 else 0.0
            avg_util = (gpu_util_samples_sum / gpu_util_samples_n) if gpu_util_samples_n else 0.0
            overall_tps = tokens_done / (sum_gen_gpu_ms / 1000) if sum_gen_gpu_ms > 0 else 0.0

            logger.info(
                f"[esg-ocr][gpu={gpu_id}][{pdf_name}] OCR SUMMARY: "
                f"{pages_done}/{len(valid_pages)} pages, "
                f"{tokens_done} tok in {ocr_total_t:.2f}s | "
                f"{pages_done / ocr_total_t:.2f} pages/s, "
                f"{overall_tps:.0f} tok/s | "
                f"breakdown: img_load {sum_img_load_ms/1000:.2f}s "
                f"({sum_img_load_ms/total_wall_ms*100 if total_wall_ms else 0:.0f}%), "
                f"generate {sum_gen_wall_ms/1000:.2f}s "
                f"(gpu {sum_gen_gpu_ms/1000:.2f}s "
                f"[{overall_gpu_pct:.0f}%], "
                f"cpu_overhead {total_cpu_overhead_ms/1000:.2f}s "
                f"[{100-overall_gpu_pct:.0f}%]), "
                f"postproc {sum_postproc_ms/1000:.2f}s "
                f"({sum_postproc_ms/total_wall_ms*100 if total_wall_ms else 0:.0f}%) | "
                f"nvidia-smi avg util {avg_util:.0f}% "
                f"(n={gpu_util_samples_n}, batch_size={batch_size})"
            )

        # Merge new OCR results into the return value
        ocr_results.update(new_ocr)

        # Update cache: merge new pages into existing cache file (rich format)
        if new_ocr:
            self._update_ocr_cache(pdf_name, new_ocr, new_tables, new_page_meta)

        # Clean up page images if configured
        if Config.CLEANUP_IMAGES_AFTER_RECOGNITION:
            recognizer._cleanup_pdf_images(pdf_name)

        logger.info(f"[{pdf_name}] OCR completed: {len(ocr_results)}/{len(page_numbers)} pages total (newly OCR'd {len(new_ocr)})")
        return ocr_results

    def _load_cached_ocr(self, cached_ocr_file: Path, page_numbers: List[int]) -> Dict[int, str]:
        """Load OCR results from existing cache file."""
        try:
            with open(cached_ocr_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

        ocr_results: Dict[int, str] = {}
        wanted = {str(p) for p in page_numbers}

        # (1) Rich numeric format: whole-page html kept in page_html map
        page_html = cached_data.get("page_html", {})
        if isinstance(page_html, dict):
            for page_num in page_numbers:
                html = page_html.get(str(page_num))
                if html:
                    ocr_results[page_num] = html

        # (2) Legacy minimal format: pages is a dict of {page: {html/markdown}}
        pages_data = cached_data.get("pages", {})
        if isinstance(pages_data, dict):
            for page_num in page_numbers:
                if page_num in ocr_results:
                    continue
                page_info = pages_data.get(str(page_num))
                if isinstance(page_info, dict):
                    content = page_info.get("html") or page_info.get("markdown")
                    if content:
                        ocr_results[page_num] = content

        # (3) Rich table list (chandra_ocr_main OR numeric rich): reconstruct page
        #     html by concatenating each page's table content_html.
        tables = cached_data.get("tables") or cached_data.get("table_blocks") or []
        if isinstance(tables, list):
            page_tables: Dict[int, List[str]] = {}
            for block in tables:
                page_key = str(block.get("page_number"))
                if page_key not in wanted:
                    continue
                html = block.get("content_html") or ""
                if html:
                    page_tables.setdefault(int(block["page_number"]), []).append(html)
            for page_num, htmls in page_tables.items():
                if page_num not in ocr_results:
                    ocr_results[page_num] = "\n".join(htmls)

        if ocr_results:
            logger.info(f"Loaded {len(ocr_results)} pages from cached OCR: {cached_ocr_file}")
        return ocr_results

    def _save_ocr_cache(self, pdf_name: str, ocr_results: Dict[int, str]) -> None:
        """Save OCR results to cache for future reuse."""
        cache_dir = Config.CHANDRA_OCR_RESULT_DIR / pdf_name
        ensure_directory_exists(cache_dir)
        cache_file = cache_dir / f"{pdf_name}_ocr_output.json"

        pages_data = {}
        for page_num, content in ocr_results.items():
            pages_data[str(page_num)] = {"html": content}

        cache_payload = {"pdf_name": pdf_name, "pages": pages_data}
        save_json(cache_payload, cache_file)
        logger.info(f"[{pdf_name}] Saved OCR cache: {cache_file}")

    def _update_ocr_cache(
        self,
        pdf_name: str,
        new_pages: Dict[int, str],
        new_tables: Optional[List[Dict[str, Any]]] = None,
        new_page_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Merge newly OCR'd pages into the cache file, in RICH format:
        - tables[]:     per-table records (table_block_id, page_number, bbox, section, html)
        - pages[]:      per-page records (page_mode, parse_quality_score, table_count)
        - page_html{}:  page_number -> whole-page html, for resume + downstream reuse
        """
        cache_dir = Config.CHANDRA_OCR_RESULT_DIR / pdf_name
        ensure_directory_exists(cache_dir)
        cache_file = cache_dir / f"{pdf_name}_ocr_output.json"

        existing_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # page_html: keep whole-page html keyed by page number (migrate old dict format)
        page_html = existing_data.get("page_html", {})
        old_pages = existing_data.get("pages", {})
        if isinstance(old_pages, dict):  # migrate legacy minimal format
            for k, v in old_pages.items():
                if isinstance(v, dict) and v.get("html") and k not in page_html:
                    page_html[k] = v["html"]
        for page_num, content in new_pages.items():
            page_html[str(page_num)] = content

        # tables: drop any prior records for re-OCR'd pages, then append new ones
        reocr = {str(p) for p in new_pages}
        tables = [t for t in existing_data.get("tables", [])
                  if str(t.get("page_number")) not in reocr]
        tables.extend(new_tables or [])

        # pages: same replace-by-page merge for the rich per-page metadata list
        old_meta = existing_data.get("pages", [])
        old_meta = old_meta if isinstance(old_meta, list) else []
        pages_meta = [p for p in old_meta if str(p.get("page_number")) not in reocr]
        pages_meta.extend(new_page_meta or [])

        cache_payload = {
            "pdf_name": pdf_name,
            "source_format_version": "numeric_extractor_rich_v1",
            "pages": pages_meta,
            "tables": tables,
            "page_html": page_html,
        }
        save_json(cache_payload, cache_file)
        logger.info(
            f"[{pdf_name}] Updated OCR cache (rich): +{len(new_pages)} pages, "
            f"{len(tables)} tables total"
        )

    def extract_numeric_blocks(
        self,
        pdf_name: str,
        pdf_path: Optional[Path] = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Main entry: extract all numeric blocks for a PDF.
        Returns combined list of text blocks + table page blocks (deduplicated by page).
        """
        if not force and self._has_existing_result(pdf_name):
            logger.info(f"[{pdf_name}] Skipping — existing result found")
            with open(self._get_output_path(pdf_name), 'r', encoding='utf-8') as f:
                return json.load(f)

        # Stage 1: text paragraphs with numbers
        text_blocks = self._extract_text_blocks(pdf_name)

        # Stage 2: table page content
        table_blocks = self._extract_table_blocks(pdf_name, pdf_path=pdf_path)

        # Deduplicate: if a page appears in both text_blocks and table_blocks,
        # keep the table_block (richer context) and drop text_blocks from that page
        table_pages = {block["page"] for block in table_blocks}
        filtered_text_blocks = [b for b in text_blocks if b["page"] not in table_pages]

        all_blocks = table_blocks + filtered_text_blocks
        all_blocks.sort(key=lambda b: (b["page"], 0 if b["source"] == "table_page" else 1))

        # Save result
        output_path = self._get_output_path(pdf_name)
        ensure_directory_exists(output_path.parent)
        save_json(all_blocks, output_path)

        logger.info(
            f"[{pdf_name}] Total numeric blocks: {len(all_blocks)} "
            f"(table_pages={len(table_blocks)}, text={len(filtered_text_blocks)})"
        )
        return all_blocks

    def process_specific_pdfs(
        self,
        pdf_paths: List[Path],
        force: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Process numeric extraction for a list of PDFs."""
        results = {}
        for pdf_path in pdf_paths:
            pdf_name = pdf_path.stem
            try:
                blocks = self.extract_numeric_blocks(pdf_name, pdf_path=pdf_path, force=force)
                results[pdf_name] = blocks
            except Exception as e:
                logger.error(f"Failed to extract numeric blocks for {pdf_name}: {e}", exc_info=True)
                results[pdf_name] = []
        return results
