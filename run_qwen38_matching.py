#!/usr/bin/env python3
"""
Run LLM variable matching for all GT PDFs using Qwen3.8-Max with reasoning.

Auto-discovers all PDFs from numeric_extracts/ (or NUMERIC_EXTRACT_DIR env var).
Outputs to quantitative_results_qwen38/ (preserves old quantitative_results_Qwen/).

Directory separation for qwen3.7 vs qwen3.8 comparison:
  OLD (qwen3.7):                         NEW (qwen3.8 + thinking):
  numeric_extracts/                      numeric_extracts_qwen38/       (if re-extracted)
  quantitative_results_Qwen/             quantitative_results_qwen38/   (always new)
  calculation_results/                   calculation_results_qwen38/    (always new)

Env vars (all have sensible defaults):
  QWEN_MAX_MODEL          default: qwen3.8-max
  ENABLE_THINKING         default: true
  THINKING_MAX_TOKENS     default: 16000
  QUANTITATIVE_RESULT_DIR default: quantitative_results_qwen38
  NUMERIC_EXTRACT_DIR     default: numeric_extracts (set to numeric_extracts_qwen38 after re-extraction)

Usage:
  # Dry run (list PDFs, estimate cost)
  python run_qwen38_matching.py --dry-run

  # Full run (all PDFs, reads from default numeric_extracts/)
  python run_qwen38_matching.py

  # Resume (skip already-processed PDFs)
  python run_qwen38_matching.py --resume

  # Read from qwen38 extraction results (after --reextract)
  python run_qwen38_matching.py --numeric-dir numeric_extracts_qwen38

  # Specific PDFs only
  python run_qwen38_matching.py --pdfs 2025041601082 2025042902036

  # Disable thinking (use qwen3.8-max without reasoning)
  python run_qwen38_matching.py --no-thinking
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- Model configuration ---
# Override defaults BEFORE importing Config so env vars take effect
os.environ.setdefault('QWEN_MAX_MODEL', 'qwen3.8-max')
os.environ.setdefault('ENABLE_THINKING', 'true')
os.environ.setdefault('THINKING_MAX_TOKENS', '16384')
os.environ.setdefault('QUANTITATIVE_RESULT_DIR', 'quantitative_results_qwen38')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.llm_variable_matcher import LLMVariableMatcher
from src.utils import setup_logger

logger = setup_logger(__name__, "logs/qwen38_matching.log")


def discover_pdfs(numeric_extract_dir: Path) -> list[str]:
    """Auto-discover all PDF names from numeric_extracts/ directory."""
    pdfs = []
    if not numeric_extract_dir.exists():
        print(f"ERROR: {numeric_extract_dir} does not exist")
        return pdfs
    for d in sorted(numeric_extract_dir.iterdir()):
        if d.is_dir() and (d / "numeric_blocks.json").exists():
            pdfs.append(d.name)
    return pdfs


def is_already_done(matcher: LLMVariableMatcher, pdf_name: str) -> bool:
    """Check if the PDF has already been processed (output file exists)."""
    output_dir = Config.QUANTITATIVE_RESULT_DIR
    output_file = output_dir / pdf_name / f"{pdf_name}_quantitative_analysis.json"
    return output_file.exists()


def main():
    parser = argparse.ArgumentParser(description='Run Qwen3.8-Max LLM matching for all GT PDFs')
    parser.add_argument('--dry-run', action='store_true',
                        help='List PDFs and estimate cost without running')
    parser.add_argument('--resume', action='store_true',
                        help='Skip PDFs that already have output')
    parser.add_argument('--pdfs', nargs='*', default=None,
                        help='Specific PDF names to process')
    parser.add_argument('--no-thinking', action='store_true',
                        help='Disable reasoning mode (use qwen3.8-max without enable_thinking)')
    parser.add_argument('--model', default=None,
                        help='Override model name (default: qwen3.8-max)')
    parser.add_argument('--numeric-dir', default=None,
                        help='Override numeric_extracts directory (default: from env or numeric_extracts)')
    args = parser.parse_args()

    # Apply overrides
    if args.no_thinking:
        os.environ['ENABLE_THINKING'] = 'false'
        # Must reload Config
        Config.ENABLE_THINKING = False
    if args.model:
        os.environ['QWEN_MAX_MODEL'] = args.model
        Config.QWEN_MAX_MODEL = args.model
    if args.numeric_dir:
        os.environ['NUMERIC_EXTRACT_DIR'] = args.numeric_dir
        Config.NUMERIC_EXTRACT_DIR = Path(args.numeric_dir)

    # Print configuration
    print("=" * 70)
    print("Qwen3.8-Max LLM Matching Runner")
    print("=" * 70)
    print(f"  Model:           {Config.QWEN_MAX_MODEL}")
    print(f"  Thinking:        {getattr(Config, 'ENABLE_THINKING', False)}")
    print(f"  Max tokens:      {getattr(Config, 'THINKING_MAX_TOKENS', 8000)}")
    print(f"  Output dir:      {Config.QUANTITATIVE_RESULT_DIR}")
    print(f"  Numeric extracts:{Config.NUMERIC_EXTRACT_DIR}")
    print(f"  API key:         {'set' if Config.QWEN_MAX_API_KEY else 'NOT SET'}")
    print("=" * 70)

    # Discover PDFs
    numeric_dir = Config.NUMERIC_EXTRACT_DIR
    if args.pdfs:
        pdf_names = args.pdfs
    else:
        pdf_names = discover_pdfs(numeric_dir)

    if not pdf_names:
        print("No PDFs found. Check numeric_extracts/ directory.")
        sys.exit(1)

    # Filter already done if --resume
    matcher = LLMVariableMatcher()
    matcher.api_key = Config.QWEN_MAX_API_KEY or os.environ.get('QWEN_MAX_API_KEY', '')
    matcher.headers = {
        "Authorization": f"Bearer {matcher.api_key}",
        "Content-Type": "application/json",
    }

    if args.resume:
        todo = [p for p in pdf_names if not is_already_done(matcher, p)]
        skipped = len(pdf_names) - len(todo)
        print(f"\nResume mode: {skipped} already done, {len(todo)} to process")
        pdf_names = todo

    print(f"\nTotal PDFs to process: {len(pdf_names)}")

    # Cost estimate
    # Average: ~10 batches/PDF, ~3K input tokens/batch, ~2K output tokens/batch
    # With thinking: ~5K thinking tokens/batch
    avg_batches = 10
    avg_input = 3000
    avg_output = 2000
    avg_thinking = 5000 if getattr(Config, 'ENABLE_THINKING', False) else 0
    est_input = len(pdf_names) * avg_batches * avg_input
    est_output = len(pdf_names) * avg_batches * (avg_output + avg_thinking)
    model = Config.QWEN_MAX_MODEL
    if "3.8" in model:
        in_price, out_price = 12.0, 36.0
    elif "3.7" in model:
        in_price, out_price = 2.0, 8.0
    else:
        in_price, out_price = 2.0, 6.0
    est_cost = est_input / 1e6 * in_price + est_output / 1e6 * out_price
    print(f"\nEstimated cost: ~¥{est_cost:.0f} ({est_input/1e6:.1f}M input + {est_output/1e6:.1f}M output tokens)")
    print(f"  Pricing: ¥{in_price}/1M input, ¥{out_price}/1M output")

    if args.dry_run:
        print("\n[DRY RUN] PDFs to process:")
        for i, p in enumerate(pdf_names, 1):
            print(f"  {i:3d}. {p}")
        print(f"\nTotal: {len(pdf_names)} PDFs")
        return

    # Confirm
    print(f"\nAbout to process {len(pdf_names)} PDFs with {model}")
    if getattr(Config, 'ENABLE_THINKING', False):
        print("  ** Reasoning mode ENABLED — responses will be slower but higher quality **")
    response = input("\nProceed? (yes/no): ")
    if response.lower() not in ('yes', 'y'):
        print("Aborted.")
        return

    # Run matching
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    results_summary = []
    start_time = time.time()

    for i, pdf_name in enumerate(pdf_names, 1):
        elapsed = time.time() - start_time
        if i > 1:
            eta = elapsed / (i - 1) * (len(pdf_names) - i + 1)
            print(f"\n[{i}/{len(pdf_names)}] {pdf_name} (elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)")
        else:
            print(f"\n[{i}/{len(pdf_names)}] {pdf_name}")

        try:
            result = matcher.match_variables_for_pdf(pdf_name, force=True)
            matches = result.get('enhancement_metadata', {}).get('total_matches', 0)
            results_summary.append((pdf_name, matches, "OK"))
            print(f"  -> {matches} matches")
        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append((pdf_name, 0, f"ERROR: {e}"))

    # Final summary
    total_elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    ok_count = 0
    for pdf_name, matches, status in results_summary:
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {pdf_name}: {matches:3d} matches  [{status}]")
        if status == "OK":
            ok_count += 1
    print(f"\n  Total PDFs: {len(pdf_names)}")
    print(f"  Successful: {ok_count}")
    print(f"  Failed:     {len(pdf_names) - ok_count}")
    print(f"  Time:       {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"  Model:      {model}")
    print(f"  Thinking:   {getattr(Config, 'ENABLE_THINKING', False)}")
    print(f"  Output dir: {Config.QUANTITATIVE_RESULT_DIR}")


if __name__ == "__main__":
    main()
