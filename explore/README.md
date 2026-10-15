# Simple Feedforward 3DGS with WorldMirror

Simplest example of running WorldMirror inference on Modal cloud GPU.

## Prerequisites

1. Set up Modal account and authenticate:
```bash
uv run --with modal modal setup
```

## Usage

### 1. Upload input images to Modal volume
```bash
uv run --with modal modal run explore/upload_data.py
```

### 2. Run inference on Modal GPU
```bash
uv run --with modal modal run explore/simple_feedforward_3dgs.py
```

**Note**: First run will take ~10-15 minutes to build the Modal image (PyTorch, gsplat, WorldMirror). Subsequent runs use the cached image and are much faster.

### 3. Download results from Modal volume
```bash
uv run --with modal modal run explore/download_results.py
```

## Output

Results will be saved to `data/output/tomato-bed-tiny/`:
- `gaussians.ply` - 3D Gaussian Splat (main output)
- `pts_from_pointmap.ply` - Point cloud

## Custom Inputs

To process different images:

```bash
# Upload custom data
uv run --with modal modal run explore/upload_data.py \
  --local-dir data/incoming/my-images \
  --remote-dir my-images

# Run inference
uv run --with modal modal run explore/simple_feedforward_3dgs.py \
  --input-dir my-images \
  --output-dir my-images

# Download results
uv run --with modal modal run explore/download_results.py \
  --remote-dir my-images \
  --local-dir data/output/my-images
```

## GPU Configuration

The script uses a T4 GPU by default (cost-optimized). To use a more powerful GPU, edit `simple_feedforward_3dgs.py` and change:

```python
gpu="T4"  # Options: "T4", "A10G", "A100"
```

## Viewing Results

Use any web-based Gaussian Splat viewer:
- [SuperSplat](https://playcanvas.com/supersplat)
- [antimatter15's viewer](https://antimatter15.com/splat/)
