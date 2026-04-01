#!/bin/bash

# Script to run cudaCC_div on all files in resampled_func_images directory
# Usage: ./process_all_images.sh [output_directory]
# 
# If no output directory is specified, will use /tmp/mocca_output

# Define paths
CUDA_PROGRAM="/mnt/islay/MOCCA/01_cudaCC/build/cudaCC_div"
INPUT_DIR="/mnt/highlands/data/MOCCA_UCLA/resampled_func_images_2mm"
MASK_FILE="/mnt/islay/MOCCA/templates/mask2mm.nii"

# Set output directory - use command line argument or default to /tmp
if [ $# -eq 1 ]; then
    OUTPUT_DIR="$1"
else
    OUTPUT_DIR="/tmp/mocca_output"
fi

# Check if the CUDA program exists
if [ ! -f "$CUDA_PROGRAM" ]; then
    echo "Error: cudaCC_div program not found at $CUDA_PROGRAM"
    echo "Please ensure the program is compiled first."
    exit 1
fi

# Check if the mask file exists
if [ ! -f "$MASK_FILE" ]; then
    echo "Error: Mask file not found at $MASK_FILE"
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found at $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
    echo "Error: Cannot create output directory $OUTPUT_DIR"
    echo "Please check disk space and permissions, or specify a different output directory."
    echo "Usage: $0 [output_directory]"
    exit 1
fi

echo "Starting batch processing..."
echo "CUDA program: $CUDA_PROGRAM"
echo "Input directory: $INPUT_DIR"
echo "Mask file: $MASK_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "----------------------------------------"

# Counter for processed files
processed=0
failed=0

# Process each .nii file in the input directory
for input_file in "$INPUT_DIR"/*.nii; do
    # Check if file exists (in case no .nii files found)
    if [ ! -f "$input_file" ]; then
        echo "No .nii files found in $INPUT_DIR"
        exit 1
    fi
    
    # Extract filename without path and extension
    filename=$(basename "$input_file" .nii)
    
    # Define output file path
    output_file="$OUTPUT_DIR/${filename}.ccmat"
    
    echo "Processing: $filename"
    echo "  Input: $input_file"
    echo "  Output: $output_file"
    
    # Run the CUDA program
    if "$CUDA_PROGRAM" "$input_file" "$MASK_FILE" "$output_file"; then
        echo "  ✓ Successfully processed $filename"
        ((processed++))
    else
        echo "  ✗ Failed to process $filename"
        ((failed++))
    fi
    
    echo "----------------------------------------"
done

# Summary
echo "========================================"
echo "Batch processing completed!"
echo "Successfully processed: $processed files"
echo "Failed: $failed files"
echo "Output files saved in: $OUTPUT_DIR"

if [ $processed -gt 0 ]; then
    echo ""
    echo "Sample output files:"
    ls -lh "$OUTPUT_DIR"/*.ccmat 2>/dev/null | head -3
fi

# Display total output size
if [ -d "$OUTPUT_DIR" ]; then
    total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "Total output size: $total_size"
fi
