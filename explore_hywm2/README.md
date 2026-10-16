# Simple Feedforward 3DGS with WorldMirror

Simplest example of running WorldMirror 2 inference on Modal cloud GPU.

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
    --dataset-tag tomato-small-2 \
    --model-tag hy-wm-2.0
```

## GPU Configuration

The script uses an A10G GPU (24GB VRAM required for 16 images). For smaller datasets (fewer images), you can use a T4 GPU by editing `simple_feedforward_3dgs.py`:

```python
gpu="A10G"  # Options: "T4" (14GB), "A10G" (24GB), "A100" (40GB)
```

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

- `hy-wm-2.0` - HunyuanWorld-Mirror 2.0 (uses `launch_inference.py`)

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
