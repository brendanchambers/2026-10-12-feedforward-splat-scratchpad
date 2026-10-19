# Simple Feedforward 3DGS with WorldMirror

Simple example running WorldMirror 2 inference on Modal cloud GPUs.


## Usage

 
convenience outer pipeline:
  # Single GPU mode (default)
  uv run --with modal modal run explore_hywm2/pipeline.py --dataset-tag tomato-small --model-tag hy-wm-2.0 --gpu-mode 1gpu

  # FSDP mode
  uv run --with modal modal run explore_hywm2/pipeline.py --dataset-tag tomato-small --model-tag hy-wm-2.0 --gpu-mode fsdp
  

## GPU Configuration

To keep costs low, use A10G for small dev and A100:80G for larger dev.  

## Data Management

### Clearing Stale Data

For a fresh run with clean data, use the `--delete-data` flag or the standalone delete script:

```bash
# Delete data as part of pipeline (recommended for fresh runs)
uv run --with modal modal run explore_hywm2/pipeline.py --dataset-tag my-images --delete-data

# Or use standalone delete script
uv run --with modal modal run explore_hywm2/delete_data.py --dataset-tag my-images
```

This removes both incoming and output data for the specified dataset tag from the Modal volume.

### Volume Synchronization

The scripts use Modal's `volume.reload()` to ensure fresh data visibility:
This prevents stale data issues when running sequential jobs.

## Viewing Results

Use any web-based Gaussian Splat viewer:
- [SuperSplat](https://playcanvas.com/supersplat)

### Supported Models

- `hy-wm-2.0` - HunyuanWorld-Mirror 2.0 (uses `launch_inference.py`)

