# Simple Feedforward 3DGS with WorldMirror

Simplest example of running WorldMirror inference on Modal cloud GPU.

## Prerequisites

1. Set up Modal account and authenticate:
```bash
uv run --with modal modal setup
```

## Usage

### Quick Start: Full Pipeline

Run the complete workflow (upload → inference → optimize → download) with a single command:

```bash
# Basic usage (reads from data/incoming/my-images/)
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images

# With custom images directory and verbose output
uv run --with modal modal run explore/pipeline.py --dataset-tag tomato-med --images-dir data/incoming/tomato-med/images --verbose

# Skip upload if data already on Modal
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --skip-upload

# Skip optimization step (faster, lower quality)
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --skip-optimize

# Custom optimization steps (default: 3000)
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --optimize-steps 5000

# Just run inference (no download)
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --skip-download
```

### Individual Steps

All scripts use `--dataset-tag` to organize outputs. By default, images are read from `data/incoming/{dataset-tag}/`, but you can specify a custom path with `--images-dir`.

```bash
# Upload data (default: from data/incoming/my-images/)
uv run --with modal modal run explore/sync_data_up.py --dataset-tag my-images

# Or use a custom images directory
uv run --with modal modal run explore/sync_data_up.py --dataset-tag my-run --images-dir path/to/images

# Run inference (includes COLMAP data for optimization)
uv run --with modal modal run explore/simple_feedforward_3dgs.py --dataset-tag my-images

# Run post-3DGS optimization (optional, improves quality)
uv run --with modal modal run explore/optimize_3dgs.py --dataset-tag my-images --max-steps 3000

# Download results (to data/output/my-images/)
uv run --with modal modal run explore/sync_data_down.py --dataset-tag my-images
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

## Viewing Results

Use any web-based Gaussian Splat viewer:
- [SuperSplat](https://playcanvas.com/supersplat)
- [antimatter15's viewer](https://antimatter15.com/splat/)

Compare the original (`gaussians.ply`) vs optimized (`optimized/saves/*.ply`) results.
