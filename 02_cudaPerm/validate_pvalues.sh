#!/bin/bash
# Validation script to check if p-values are reasonable after bug fix

echo "================================================"
echo "P-Value Validation Script"
echo "================================================"
echo ""

if [ $# -ne 2 ]; then
    echo "Usage: $0 <output_file.permout> <number_of_permutations>"
    echo ""
    echo "Example: $0 output.permout 5000"
    exit 1
fi

OUTPUT_FILE=$1
NUM_PERM=$2

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Error: File not found: $OUTPUT_FILE"
    exit 1
fi

echo "Analyzing: $OUTPUT_FILE"
echo "Expected number of permutations: $NUM_PERM"
echo ""

# Calculate expected minimum p-value
MIN_EXPECTED=$(echo "scale=6; 1.0 / ($NUM_PERM + 1)" | bc)
echo "Expected minimum p-value: $MIN_EXPECTED"
echo ""

# Count total values
TOTAL_VALUES=$(awk '{for(i=1;i<=NF;i++) if($i!="") count++} END{print count}' "$OUTPUT_FILE")
echo "Total p-values in file: $TOTAL_VALUES"
echo ""

# Check for exact zeros (should be very rare or none)
EXACT_ZEROS=$(grep -o "0\.000000 " "$OUTPUT_FILE" | wc -l)
echo "Exact zero p-values (0.000000): $EXACT_ZEROS"
if [ $EXACT_ZEROS -gt 0 ]; then
    ZERO_PERCENT=$(echo "scale=2; 100.0 * $EXACT_ZEROS / $TOTAL_VALUES" | bc)
    echo "  → That's $ZERO_PERCENT% of all values"
    if (( $(echo "$ZERO_PERCENT > 5" | bc -l) )); then
        echo "  ⚠️  WARNING: More than 5% zeros - this seems unusual!"
    fi
fi
echo ""

# Check for exact ones (should be very rare)
EXACT_ONES=$(grep -o "1\.000000 " "$OUTPUT_FILE" | wc -l)
echo "Exact one p-values (1.000000): $EXACT_ONES"
if [ $EXACT_ONES -gt 0 ]; then
    ONE_PERCENT=$(echo "scale=2; 100.0 * $EXACT_ONES / $TOTAL_VALUES" | bc)
    echo "  → That's $ONE_PERCENT% of all values"
fi
echo ""

# Get min and max p-values
echo "P-value range:"
MIN_ACTUAL=$(awk '{for(i=1;i<=NF;i++) if($i!="") print $i}' "$OUTPUT_FILE" | sort -n | head -1)
MAX_ACTUAL=$(awk '{for(i=1;i<=NF;i++) if($i!="") print $i}' "$OUTPUT_FILE" | sort -n | tail -1)
echo "  Minimum: $MIN_ACTUAL (expected: ~$MIN_EXPECTED)"
echo "  Maximum: $MAX_ACTUAL (expected: ~1.0)"
echo ""

# Check if minimum is reasonable
if (( $(echo "$MIN_ACTUAL < $MIN_EXPECTED" | bc -l) )); then
    echo "⚠️  WARNING: Minimum p-value is smaller than expected!"
    echo "    This suggests an error in the calculation."
fi

if (( $(echo "$MIN_ACTUAL > $(echo "scale=6; 5.0 / ($NUM_PERM + 1)" | bc)" | bc -l) )); then
    echo "⚠️  NOTE: Minimum p-value is larger than 5×expected."
    echo "    This might be fine if your data has no strong effects."
fi

if (( $(echo "$MAX_ACTUAL > 1.0" | bc -l) )); then
    echo "❌ ERROR: Maximum p-value exceeds 1.0!"
    echo "    This indicates a serious bug in the calculation."
fi

# Distribution check
echo ""
echo "P-value distribution (deciles):"
awk '{for(i=1;i<=NF;i++) if($i!="") print $i}' "$OUTPUT_FILE" | sort -n | awk '
    BEGIN {count=0}
    {values[count++] = $1}
    END {
        for(i=0; i<=10; i++) {
            idx = int(count * i / 10)
            if (idx >= count) idx = count - 1
            printf "  %3d%%: %f\n", i*10, values[idx]
        }
    }
'

echo ""
echo "Under null hypothesis (no real effect), p-values should be"
echo "uniformly distributed between 0 and 1."
echo ""

# Check for significant results
SIGNIFICANT_005=$(awk '{for(i=1;i<=NF;i++) if($i!="" && $i<0.05) count++} END{print count+0}' "$OUTPUT_FILE")
SIGNIFICANT_001=$(awk '{for(i=1;i<=NF;i++) if($i!="" && $i<0.01) count++} END{print count+0}' "$OUTPUT_FILE")
SIGNIFICANT_0001=$(awk '{for(i=1;i<=NF;i++) if($i!="" && $i<0.001) count++} END{print count+0}' "$OUTPUT_FILE")

SIG_PERCENT_005=$(echo "scale=2; 100.0 * $SIGNIFICANT_005 / $TOTAL_VALUES" | bc)
SIG_PERCENT_001=$(echo "scale=2; 100.0 * $SIGNIFICANT_001 / $TOTAL_VALUES" | bc)
SIG_PERCENT_0001=$(echo "scale=2; 100.0 * $SIGNIFICANT_0001 / $TOTAL_VALUES" | bc)

echo "Significant results:"
echo "  p < 0.05:  $SIGNIFICANT_005 ($SIG_PERCENT_005%)"
echo "  p < 0.01:  $SIGNIFICANT_001 ($SIG_PERCENT_001%)"
echo "  p < 0.001: $SIGNIFICANT_0001 ($SIG_PERCENT_0001%)"
echo ""

# Summary
echo "================================================"
echo "Summary"
echo "================================================"
if [ $EXACT_ZEROS -eq 0 ] && (( $(echo "$MIN_ACTUAL >= $MIN_EXPECTED" | bc -l) )) && (( $(echo "$MAX_ACTUAL <= 1.0" | bc -l) )); then
    echo "✅ P-values look reasonable!"
    echo "   - No exact zeros"
    echo "   - Range is appropriate"
    echo "   - Minimum p-value matches expected"
else
    echo "⚠️  Some concerns detected. Please review the warnings above."
fi
echo ""
