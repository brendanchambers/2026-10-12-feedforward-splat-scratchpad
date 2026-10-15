# Simple Feedforward 3DGS with WorldMirror

Simplest example of running WorldMirror inference on Modal cloud GPU.

## Prerequisites

1. Set up Modal account and authenticate:
```bash
uv run --with modal modal setup
```

## Usage

### Quick Start: Full Pipeline

Run the complete workflow (upload → inference → download) with a single command:

```bash
# Basic usage (reads from data/incoming/my-images/)
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images

# With custom images directory
uv run --with modal modal run explore/pipeline.py --dataset-tag my-run --images-dir path/to/images

# Skip upload if data already on Modal
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --skip-upload

# Just run inference (no download)
uv run --with modal modal run explore/pipeline.py --dataset-tag my-images --skip-download
```

### Individual Steps

All scripts use `--dataset-tag` to organize outputs. By default, images are read from `data/incoming/{dataset-tag}/`, but you can specify a custom path with `--images-dir`.

```bash
# Upload data (default: from data/incoming/my-images/)
uv run --with modal modal run explore/upload_data.py --dataset-tag my-images

# Or use a custom images directory
uv run --with modal modal run explore/upload_data.py --dataset-tag my-run --images-dir path/to/images

# Run inference
uv run --with modal modal run explore/simple_feedforward_3dgs.py --dataset-tag my-images

# Download results (to data/output/my-images/)
uv run --with modal modal run explore/download_results.py --dataset-tag my-images
```

## GPU Configuration

The script uses an A10G GPU (24GB VRAM required for 16 images). For smaller datasets (fewer images), you can use a T4 GPU by editing `simple_feedforward_3dgs.py`:

```python
gpu="A10G"  # Options: "T4" (14GB), "A10G" (24GB), "A100" (40GB)
```

## Viewing Results

Use any web-based Gaussian Splat viewer:
- [SuperSplat](https://playcanvas.com/supersplat)
- [antimatter15's viewer](https://antimatter15.com/splat/)
