# Simple Feedforward 3DGS with WorldMirror

Simplest example of running WorldMirror inference on Modal cloud GPU.

## Prerequisites

1. Set up Modal account and authenticate:
```bash
uv run --with modal modal setup
```

## Usage

### Quick Start: Full Pipeline

Run the complete workflow (upload → inference → optimize → download) with a single command.

**Recommended usage** (fresh run with optimization):
```bash
uv run --with modal modal run explore/pipeline.py \
  --dataset-tag tomato-small \
  --images-dir data/incoming/tomato-small/images \
  --optimize-steps 500 \
  --delete-data \
  --verbose
```

This command:
- Deletes any existing data for `tomato-small` (fresh start)
- Uploads images from custom directory
- Runs feedforward inference with verbose output
- Runs post-processing optimization (500 steps)
- Downloads results to `data/output/tomato-small/`

**Fast iteration** (skip optimization, keep data):
```bash
uv run --with modal modal run explore/pipeline.py \
  --dataset-tag tomato-tiny \
  --images-dir data/incoming/tomato-tiny/images \
  --skip-optimize \
  --verbose
```

This command:
- Uses existing data if available (no deletion)
- Uploads from default location `data/incoming/tomato-small/`
- Runs feedforward inference only (faster)
- Downloads results immediately
- Good for quick testing and iteration

**Other common patterns**:
```bash

# skips
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --skip-upload --skip-optimize --skip-download

### Individual Steps

All scripts use `--dataset-tag` to organize outputs. By default, images are read from `data/incoming/{dataset-tag}/`, but you can specify a custom path with `--images-dir`.

```bash
# Upload data (default: from data/incoming/my-images/)
uv run --with modal modal run explore/upload_data.py --dataset-tag my-images

# Or use a custom images directory
uv run --with modal modal run explore/upload_data.py --dataset-tag my-run --images-dir path/to/images

# Run inference (includes COLMAP data for optimization)
uv run --with modal modal run explore/simple_feedforward_3dgs.py --dataset-tag my-images

# Run post-3DGS optimization (optional, improves quality)
uv run --with modal modal run explore/optimize_3dgs.py --dataset-tag my-images --max-steps 3000

# Download results (to data/output/my-images/)
uv run --with modal modal run explore/download_results.py --dataset-tag my-images
```

## GPU Configuration

The script uses an A10G GPU (24GB VRAM required for 16 images). For smaller datasets (fewer images), you can use a T4 GPU by editing `simple_feedforward_3dgs.py`:

```python
gpu="A10G"  # Options: "T4" (14GB), "A10G" (24GB), "A100" (40GB)
```

## Post-3DGS Optimization

The pipeline includes an optional post-processing optimization step that refines the initial 3DGS output for improved quality. This uses the gsplat trainer on the COLMAP data generated during inference.

**What it does:**
- Refines Gaussian parameters (positions, scales, opacities, colors)
- Uses the original input images and camera parameters
- Runs for 3000 steps by default (configurable)

**Output structure:**
```
data/output/{dataset-tag}/
├── gaussians.ply              # Original feedforward 3DGS
├── images/                    # Processed input images
├── sparse/                    # COLMAP reconstruction data
└── optimized/                 # Post-optimization results
    ├── ckpts/                 # Training checkpoints
    └── saves/                 # Final optimized .ply files
```

**When to use:**
- Use optimization for highest quality results
- Skip with `--skip-optimize` for faster iteration during development
- Adjust `--optimize-steps` to balance quality vs. speed

## Data Management

### Clearing Stale Data

For a fresh run with clean data, use the `--delete-data` flag or the standalone delete script:

```bash
# Delete data as part of pipeline (recommended for fresh runs)
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --delete-data

# Or use standalone delete script
uv run --with modal modal run explore/delete_data.py --dataset-tag my-images
```

This removes both incoming and output data for the specified dataset tag from the Modal volume.

### Volume Synchronization

The scripts use Modal's `volume.reload()` to ensure fresh data visibility:
- **Upload → Inference**: Inference reloads to see uploaded images
- **Inference → Optimization**: Optimization reloads to see inference outputs
- **Optimization → Download**: Download reloads to see optimized results

This prevents stale data issues when running sequential jobs.

## Viewing Results

Use any web-based Gaussian Splat viewer:
- [SuperSplat](https://playcanvas.com/supersplat)
- [antimatter15's viewer](https://antimatter15.com/splat/)

Compare the original (`gaussians.ply`) vs optimized (`optimized/saves/*.ply`) results.

## After Refactor

The pipeline now supports multiple model versions via the `--model-tag` parameter. This allows you to easily switch between different inference models.

### Supported Models

- `hy-wm-1.1` - HunyuanWorld-Mirror 1.1 (uses `inference_hy-wm-1p1.py`)
- `hy-wm-2.0` - HunyuanWorld-Mirror 2.0 (uses `hy_wm_2_0_inference.py`)

### Usage with Model Tags

**Run complete pipeline with specific model:**
```bash
uv run --with modal modal run explore/pipeline.py \
  --dataset-tag tomato-small-2 \
  --model-tag hy-wm-1.1
```

**Skip upload/download for faster iteration:**
```bash
uv run --with modal modal run explore/pipeline.py \
  --dataset-tag tomato-small-2 \
  --model-tag hy-wm-1.1 \
  --skip-upload \
  --skip-download
```

**Test with verbose output:**
```bash
uv run --with modal modal run explore/pipeline.py \
  --dataset-tag tomato-small-2 \
  --model-tag hy-wm-1.1 \
  --verbose
```

### Output Structure

Results are now organized by model version:
```
data/output/
├── hy-wm-1.1/
│   └── tomato-small-2/
│       ├── images/
│       ├── sparse/
│       └── gaussians.ply
└── hy-wm-2.0/
    └── tomato-small-2/
        ├── images/
        ├── sparse/
        └── gaussians.ply
```

This allows easy comparison between different model versions on the same dataset.
