#!/bin/bash
# Fast threshold counting using awk (much faster for simple operations)
# Use this for quick checks. Use Python script for comprehensive analysis.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <permout_file> [threshold1] [threshold2] ..."
    echo ""
    echo "Examples:"
    echo "  $0 results.permout 0.05"
    echo "  $0 results.permout 0.05 0.01 0.001"
    echo ""
    echo "Default thresholds: 0.05 0.01 0.001 0.0005 0.0001"
    exit 1
fi

PERMOUT_FILE=$1
shift

# Use default thresholds if none provided
if [ $# -eq 0 ]; then
    THRESHOLDS=(0.05 0.01 0.001 0.0005 0.0001)
else
    THRESHOLDS=("$@")
fi

if [ ! -f "$PERMOUT_FILE" ]; then
    echo "ERROR: File not found: $PERMOUT_FILE"
    exit 1
fi

echo "=========================================="
echo "Quick P-value Threshold Counter"
echo "=========================================="
echo "File: $PERMOUT_FILE"
echo "File size: $(du -h "$PERMOUT_FILE" | cut -f1)"
echo ""

# Count total values (quick)
echo "Counting total p-values..."
TOTAL=$(awk '{for(i=1;i<=NF;i++) if($i!="") count++} END{print count}' "$PERMOUT_FILE")
echo "Total p-values: $TOTAL"
echo ""

# Count for each threshold (single pass)
echo "Counting values below thresholds..."
echo ""

# Build awk script for all thresholds in one pass
AWK_SCRIPT='{'
AWK_SCRIPT+='for(i=1;i<=NF;i++) if($i!="") {'
for i in "${!THRESHOLDS[@]}"; do
    AWK_SCRIPT+="if(\$i<${THRESHOLDS[$i]}) count$i++; "
done
AWK_SCRIPT+='}'
AWK_SCRIPT+='} END {'
for i in "${!THRESHOLDS[@]}"; do
    AWK_SCRIPT+="print count$i+0; "
done
AWK_SCRIPT+='}'

# Run single pass
COUNTS=$(awk "$AWK_SCRIPT" "$PERMOUT_FILE")

# Display results
echo "Threshold    Count              Percentage"
echo "----------   ----------------   -----------"
idx=0
while IFS= read -r count; do
    thresh="${THRESHOLDS[$idx]}"
    percent=$(echo "scale=4; 100.0 * $count / $TOTAL" | bc)
    printf "p < %-8s %'16d   %10.4f%%\n" "$thresh" "$count" "$percent"
    idx=$((idx + 1))
done <<< "$COUNTS"

echo ""
echo "=========================================="
echo "Note: This is a fast approximation."
echo "For percentile thresholds and distribution,"
echo "use: find_pvalue_threshold.py"
echo "=========================================="
