#!/bin/bash
# Compare OCR speed between 2 GPU jobs after both finish.
# Usage: bash compare_gpu_speed.sh <job_id_1> <job_id_2>
#
# Each job should have run run_5stream_p1.sh, producing logs like:
#   logs/ocr_list_XX_job<JOB_ID>.log
#
# This script greps for [esg-ocr] telemetry lines and prints a side-by-side
# comparison of throughput, tokens/s, GPU util, CPU overhead.

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <job_id_1> <job_id_2>"
    echo "Example: $0 12345 12346"
    exit 1
fi

JOB1=$1
JOB2=$2

echo "=========================================="
echo "GPU Speed Comparison: Job $JOB1 vs Job $JOB2"
echo "=========================================="
echo ""

# Find all log files for each job
LOGS1=$(ls logs/ocr_list_*_job${JOB1}.log 2>/dev/null | sort)
LOGS2=$(ls logs/ocr_list_*_job${JOB2}.log 2>/dev/null | sort)

if [ -z "$LOGS1" ] || [ -z "$LOGS2" ]; then
    echo "ERROR: Could not find log files for one or both jobs."
    echo "Job $JOB1 logs: $LOGS1"
    echo "Job $JOB2 logs: $LOGS2"
    exit 1
fi

echo "Job $JOB1 has $(echo "$LOGS1" | wc -l) log files"
echo "Job $JOB2 has $(echo "$LOGS2" | wc -l) log files"
echo ""

# Extract OCR SUMMARY lines (one per PDF)
echo "=========================================="
echo "Per-PDF Summary (OCR SUMMARY lines)"
echo "=========================================="
echo ""

echo "--- Job $JOB1 ---"
for log in $LOGS1; do
    grep "OCR SUMMARY" "$log" 2>/dev/null || true
done | head -20

echo ""
echo "--- Job $JOB2 ---"
for log in $LOGS2; do
    grep "OCR SUMMARY" "$log" 2>/dev/null || true
done | head -20

echo ""
echo "=========================================="
echo "Aggregate Statistics"
echo "=========================================="
echo ""

# Count total pages, tokens, time for each job
for job in $JOB1 $JOB2; do
    echo "Job $job:"
    LOGS=$(ls logs/ocr_list_*_job${job}.log 2>/dev/null)

    # Sum pages
    PAGES=$(for log in $LOGS; do grep "OCR SUMMARY" "$log" 2>/dev/null | grep -oE "[0-9]+/[0-9]+ pages" | awk -F'/' '{print $1}'; done | awk '{s+=$1} END {print s+0}')
    echo "  Total pages: $PAGES"

    # Sum tokens
    TOKENS=$(for log in $LOGS; do grep "OCR SUMMARY" "$log" 2>/dev/null | grep -oE "[0-9]+ tok" | awk '{print $1}'; done | awk '{s+=$1} END {print s+0}')
    echo "  Total tokens: $TOKENS"

    # Sum time
    TIME=$(for log in $LOGS; do grep "OCR SUMMARY" "$log" 2>/dev/null | grep -oE "in [0-9.]+s" | awk '{print $2}' | sed 's/s//'; done | awk '{s+=$1} END {print s+0}')
    echo "  Total time: ${TIME}s"

    # Avg pages/s
    if [ "$(echo "$TIME > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        PPS=$(echo "scale=2; $PAGES / $TIME" | bc -l 2>/dev/null || echo "n/a")
        TPS=$(echo "scale=0; $TOKENS / $TIME" | bc -l 2>/dev/null || echo "n/a")
        echo "  Avg pages/s: $PPS"
        echo "  Avg tokens/s: $TPS"
    else
        echo "  Avg pages/s: n/a"
        echo "  Avg tokens/s: n/a"
    fi

    # GPU util (average across all SUMMARY lines)
    UTIL=$(for log in $LOGS; do grep "OCR SUMMARY" "$log" 2>/dev/null | grep -oE "avg util [0-9]+" | awk '{print $3}'; done | awk '{s+=$1; n++} END {if(n>0) print s/n; else print 0}')
    echo "  Avg GPU util: ${UTIL}%"

    # CPU overhead (average)
    CPU_OVERHEAD=$(for log in $LOGS; do grep "OCR SUMMARY" "$log" 2>/dev/null | grep -oE "cpu_overhead [0-9.]+s \[[0-9]+%\]" | grep -oE "\[[0-9]+%\]" | tr -d '[%]'; done | awk '{s+=$1; n++} END {if(n>0) print s/n; else print 0}')
    echo "  Avg CPU overhead: ${CPU_OVERHEAD}%"

    echo ""
done

echo "=========================================="
echo "Comparison Complete"
echo "=========================================="
