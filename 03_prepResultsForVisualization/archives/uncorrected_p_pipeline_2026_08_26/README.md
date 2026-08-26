# Permutation Results to CSV Converter

## Overview

The `permout_to_csv.py` script converts permutation test results (`.permout` files) to CSV format, extracting only connections that pass a significance threshold.

**Key Features:**
- ✅ **Adaptive memory management**: Automatically chooses optimal strategy based on file size
- ✅ **Simplified output**: Only coordinates (i,j,k) and statistics - no redundant indices
- ✅ **Flexible masking**: Specify custom brain templates with `-m` option
- ✅ **Progress bar**: Visual progress tracking with `tqdm` (install with `pip install tqdm`)
- ✅ **Large file support**: Handles 200+ GB files efficiently

## Quick Start

### Basic Usage

```bash
# Convert with default settings (p < 0.05)
./permout_to_csv.py ../02_cudaPerm/LTLEvsRTLE_run1.permout
```

### Common Usage Patterns

```bash
# Stricter threshold (p < 0.01)
./permout_to_csv.py ../02_cudaPerm/LTLEvsRTLE_run1.permout -t 0.01

# Custom output file
./permout_to_csv.py ../02_cudaPerm/LTLEvsRTLE_run1.permout -o results.csv

# Use different brain template
./permout_to_csv.py ../02_cudaPerm/results.permout -m ../templates/brain2p5mm.dump
```

### Running Large Files in Background

```bash
# Install tqdm for progress bar
pip install tqdm

# Run in background with logging
nohup python3 -u permout_to_csv.py \
    ../02_cudaPerm/LTLEvsRTLE_run1.permout \
    -t 0.05 \
    -o LTLEvsRTLE_significant.csv \
    > conversion.log 2>&1 &

# Monitor progress
tail -f conversion.log
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `permout_file` | Path to .permout file (required) | - |
| `-t`, `--threshold` | P-value significance threshold | 0.05 |
| `-m`, `--mask` | Brain template/coordinate file | `../templates/brain2mm.dump` |
| `--tstat` | T-statistic file path | Auto-detect |
| `-o`, `--output` | Output CSV file | Auto-generate |
| `--buffer-size` | Total buffer size (GB) for memory management | 10.0 |

## Input Files

### Required

- **P-value file** (`.permout`): Permutation test results in upper triangular format
- **Mask/template file**: Voxel coordinates (i,j,k in first 3 columns)

### Optional

- **T-statistic file** (`_tstat.permout`): Auto-detected by default

## Output Format

CSV file with columns:

| Column | Description |
|--------|-------------|
| `i1`, `j1`, `k1` | Coordinates of first voxel |
| `i2`, `j2`, `k2` | Coordinates of second voxel |
| `pvalue` | Statistical p-value |
| `tstat` | T-statistic (if available) |

**Example output:**

```csv
i1,j1,k1,i2,j2,k2,pvalue,tstat
41,36,1,38,37,1,0.023451,3.245678
41,36,1,39,37,1,0.018934,4.123456
38,38,1,41,37,1,0.042123,-2.876543
```

## Adaptive Memory Management

The script automatically chooses the optimal processing strategy based on file sizes using a **unified buffer allocation system**.

### Buffer Allocation Strategy (Updated March 2026)

The script uses a configurable buffer size (default: 10 GB) that is intelligently allocated:

**When only p-values are present:**
- Full buffer available for p-value file
- File < buffer → Load into memory (faster)
- File ≥ buffer → Stream from disk (lower memory)

**When both p-values and t-statistics are present:**
- Buffer split **50/50** between the two files
- Example: 10 GB buffer → 5 GB for p-values, 5 GB for t-stats
- Each file independently decides to load or stream based on its threshold

### Loading Decision Logic

#### Scenario 1: Only P-values (No T-statistics)
```
File size < buffer_size → Load into memory
File size ≥ buffer_size → Stream from disk
```

**Example with 10 GB buffer:**
- 8 GB p-value file → Loaded into memory ✓
- 15 GB p-value file → Streamed from disk ✓

#### Scenario 2: Both P-values and T-statistics
```
P-value threshold = buffer_size / 2
T-stat threshold = buffer_size / 2

Each file loads if below its threshold, streams otherwise
```

**Example with 10 GB buffer (5 GB each):**
- 4 GB p-values + 3 GB t-stats → Both loaded into memory ✓
- 8 GB p-values + 3 GB t-stats → P-values streamed, t-stats loaded ✓
- 8 GB p-values + 7 GB t-stats → Both streamed ✓

### Memory Strategies

#### Load into Memory
- **Pros**: Fastest processing (random access)
- **Cons**: Requires RAM ≈ file size
- **Best for**: Smaller datasets, systems with sufficient RAM

#### Stream from Disk
- **Pros**: Minimal memory (~500 MB), handles any file size
- **Cons**: Slightly slower due to I/O operations
- **Best for**: Large datasets (100+ GB), memory-constrained systems

### Adjusting Buffer Size

You can customize the buffer size to match your available RAM:

```bash
# Default (10 GB buffer)
./permout_to_csv.py input.permout

# Large memory system (20 GB buffer)
./permout_to_csv.py input.permout --buffer-size 20.0
# With t-stats: 10 GB for p-values, 10 GB for t-stats

# Low memory system (4 GB buffer)
./permout_to_csv.py input.permout --buffer-size 4.0
# With t-stats: 2 GB for p-values, 2 GB for t-stats

# Force streaming for very large files (1 GB buffer)
./permout_to_csv.py input.permout --buffer-size 1.0
```

### Key Benefits

1. ✅ **Fair Resource Allocation**: Both p-values and t-stats treated equally
2. ✅ **Memory Efficiency**: No hardcoded assumptions about file sizes
3. ✅ **Flexibility**: User controls total memory usage
4. ✅ **Automatic Optimization**: Script chooses best strategy per file
5. ✅ **Handles Any Size**: Can process arbitrarily large files

## Performance Estimates

### Example: LTLEvsRTLE_run1.permout (220 GB, 248,633 voxels)

- **Connections**: 30,909,060,028 (~31 billion)
- **Processing time**: 8-12 hours
- **Memory usage**:
  - With t-stats (202 GB): ~500 MB (streaming)
  - Without t-stats: ~500 MB
- **Expected output (p < 0.05)**: ~1.5 billion connections (~150-200 GB CSV)

### Example: controlsVSpatients_runAll.permout (5.8 GB, ~35,000 voxels)

- **Connections**: ~612 million
- **Processing time**: 20-40 minutes
- **Memory usage**: ~6 GB (loads t-stats into memory)
- **Expected output (p < 0.05)**: ~30 million connections (~3-5 GB CSV)

## Monitoring Progress

The script uses `tqdm` for a visual progress bar:

```
Processing: |████████████████████░░░░| 120000/248633 [02:15<02:30]  significant: 1,234,567
```

If `tqdm` is not installed, it falls back to simple text updates every 10,000 voxels.

```bash
# Check if process is running
ps aux | grep permout_to_csv

# View log file
tail -f conversion.log

# Count output lines (connections found so far)
wc -l output.csv

# Check output file size
ls -lh output.csv
```

## Troubleshooting

### ERROR: Unexpected end of file

**Problem**: File is corrupted or doesn't match expected format.

**Solution**: Verify that:
1. The `.permout` file is complete (not truncated)
2. The mask file matches the brain used for analysis
3. Files weren't corrupted during transfer

### High memory usage

**Problem**: Files loaded into memory despite large size.

**Solution**: Reduce buffer size to force streaming:

```bash
./permout_to_csv.py input.permout --buffer-size 1.0
```

### Too many significant connections

**Problem**: Threshold too lenient.

**Solution**: Use stricter threshold:

```bash
./permout_to_csv.py input.permout -t 0.01  # or 0.001
```

### Process killed / Out of memory

**Problem**: System ran out of memory.

**Solution**: Lower the buffer size or add more swap space:

```bash
# Force streaming for all files (minimal memory)
./permout_to_csv.py input.permout --buffer-size 0.5
```

## Technical Details

### Upper Triangular Matrix Storage

The `.permout` files store N×N connectivity matrices in upper triangular format to save space:

```
Row 0: N-1 values (voxel 0 → voxels 1, 2, ..., N-1)
Row 1: N-2 values (voxel 1 → voxels 2, 3, ..., N-1)
...
Row i: N-i-1 values (voxel i → voxels i+1, i+2, ..., N-1)
```

Total connections = N × (N-1) / 2

### Memory-Efficient Streaming

The script uses a buffered streaming reader that:

1. Reads file line by line
2. Splits each line into values
3. Returns values one at a time on demand
4. Never stores more than one line in memory

This allows processing arbitrarily large files with minimal RAM.

### Coordinate System

Coordinates from the mask file (e.g., `brain2mm.dump`) are in **voxel space** (i,j,k indices):

- `brain2mm.dump`: 2mm isotropic resolution
- `brain2p5mm.dump`: 2.5mm isotropic resolution (if available)

To convert to world coordinates (mm), use AFNI or other neuroimaging tools.

## Examples

### Example 1: Quick exploration (lenient threshold)

```bash
./permout_to_csv.py ../02_cudaPerm/LTLEvsRTLE_run1.permout -t 0.10
```

### Example 2: Final analysis (strict threshold)

```bash
./permout_to_csv.py \
    ../02_cudaPerm/controlsVSpatients_runAll.permout \
    -t 0.001 \
    -o controls_vs_patients_p0001.csv
```

### Example 3: Custom template

```bash
./permout_to_csv.py \
    ../02_cudaPerm/results.permout \
    -m ../templates/custom_mask.dump \
    -t 0.05
```

### Example 4: Force streaming (low memory mode)

```bash
./permout_to_csv.py \
    ../02_cudaPerm/large_file.permout \
    --buffer-size 1.0
```

## Migration from Old Scripts

If you were using old versions of the scripts:

1. **Old scripts backed up**: `.old` extension
2. **New unified script**: `permout_to_csv.py`
3. **Key changes**:
   - Output format simplified (no voxel indices)
   - Use `-m` instead of `-c` for mask/template
   - Automatic strategy selection (no separate "optimized" version)
   - Better progress reporting

## Recent Updates

### March 2026: Unified Buffer Strategy

**Major Update**: Both p-values and t-statistics now use the same adaptive loading strategy with shared buffer allocation.

**Breaking Changes:**
- ⚠️ Parameter renamed: `--tstat-memory-threshold` → `--buffer-size`
- ⚠️ Behavior change: Buffer now applies to both p-values and t-statistics
- Update any scripts using the old parameter name

**New Features:**
- P-values can now be loaded into memory for faster processing (small files)
- Fair 50/50 buffer split when both files are present
- Better memory efficiency for mixed file sizes
- Single parameter (`--buffer-size`) controls all memory allocation

**Migration Guide:**

```bash
# Old command (will fail)
./permout_to_csv.py input.permout --tstat-memory-threshold 20.0

# New command (correct)
./permout_to_csv.py input.permout --buffer-size 20.0
```

**Performance Impact:**
- Small p-value files now load faster (no longer always streaming)
- Large file handling unchanged (still streams efficiently)
- Memory usage more predictable and controllable

## See Also

- `../02_cudaPerm/README.md` - Permutation testing documentation
- `../02_cudaPerm/TSTAT_OUTPUT_FEATURE.md` - T-statistic format details
- `afni/` directory - Visualization tools for significant connections
- `cudaPerm/` directory - Additional permutation result processing tools

