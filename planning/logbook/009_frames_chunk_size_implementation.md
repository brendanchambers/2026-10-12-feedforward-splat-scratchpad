# frames_chunk_size Implementation

**Date**: 2026-10-19
**Purpose**: Enable user control over memory optimization via frame chunking

## Summary

Implemented end-to-end support for configurable `frames_chunk_size` parameter, allowing users to control how many frames are processed at once in dense heads. This addresses OOM issues with large image batches by processing frames in smaller chunks.

## Default Value: 1

Set to the most conservative value for maximum memory safety. Users can increase if they have more GPU memory available.

## Implementation Path

### 1. Pipeline Layer (`pipeline.py`)

**Added parameter to `__call__` method** (line ~469):
```python
def __call__(
    self,
    ...
    # Memory optimization
    frames_chunk_size: int = 1,
    ...
)
```

**Pass to `_run_inference`** (line ~521):
```python
predictions, imgs, infer_time = self._run_inference(
    img_paths, effective, prior_cam_path, prior_depth_path, frames_chunk_size
)
```

**Updated `_run_inference` signature** (line ~629):
```python
def _run_inference(self, img_paths, target_size, prior_cam_path, prior_depth_path, frames_chunk_size):
```

**Pass to model forward** (line ~679):
```python
fwd_kw = dict(views=views, cond_flags=cond_flags, is_inference=True, frames_chunk_size=frames_chunk_size)
```

### 2. Model Layer (`worldmirror.py`)

**Updated `forward` signature** (line ~255):
```python
def forward(
    self,
    views: Dict[str, torch.Tensor],
    cond_flags: List[int] = [0, 0, 0],
    is_inference=True,
    sp_size=1,
    sp_group=None,
    frames_chunk_size=1,
):
```

**Pass to prediction generation methods** (line ~294, 300):
```python
# Sequence parallel path
preds = self._gen_all_preds_frame_sp(
    token_list, imgs, patch_start_idx, views, cond_flags,
    is_inference, sp_size, sp_group, frames_chunk_size,
)

# Single GPU path
preds = self._gen_all_preds(
    token_list, imgs, patch_start_idx, views, cond_flags, is_inference, frames_chunk_size,
)
```

**Updated `_gen_all_preds_frame_sp` signature** (line ~306):
```python
def _gen_all_preds_frame_sp(
    self, token_list, imgs, patch_start_idx, views, cond_flags, is_inference,
    sp_size, sp_group, frames_chunk_size=1,
):
```

**Updated `_gen_all_preds` signature** (line ~582):
```python
def _gen_all_preds(
    self, token_list, imgs, patch_start_idx, views, cond_flags, is_inference, frames_chunk_size=1
):
```

**Pass to all head calls**:
- `depth_head` (lines ~389, 396, 600, 606)
- `pts_head` (lines ~418, 613)
- `norm_head` (lines ~432, 620)
- `gs_head` (lines ~477, 485, 633, 641)

Example:
```python
gs_feat_p, gs_depth_p, gs_depth_conf_p = self.gs_head(
    gs_token_chunk,
    images=gs_imgs_chunk,
    patch_start_idx=patch_start_idx,
    frames_chunk_size=frames_chunk_size,  # ← Added
)
```

### 3. CLI Layer (`pipeline.py`)

**Added argument parser** (line ~801):
```python
parser.add_argument("--frames_chunk_size", type=int, default=1,
                    help="Process frames in chunks to reduce memory usage (default: 1)")
```

**Added to call_kwargs** (line ~849):
```python
call_kwargs = dict(
    ...
    frames_chunk_size=args.frames_chunk_size,
    ...
)
```

## Usage

### Via CLI:
```bash
torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
    --input_path /path/to/images \
    --use_fsdp \
    --frames_chunk_size 1  # ← New parameter
```

### Via Python API:
```python
pipeline = WorldMirrorPipeline.from_pretrained("tencent/HY-World-2.0")
result = pipeline(
    "path/to/images",
    frames_chunk_size=1,  # ← New parameter
)
```

### Current launch_inference.py:
The parameter is already passed through via `--no_save_*` flags, so no changes needed to `launch_inference.py`. The CLI argument will be automatically available.

## How It Works

### Before (Default frames_chunk_size=8):
With 62 images per GPU:
- Chunk 1: frames 0-7 (8 frames)
- Chunk 2: frames 8-15 (8 frames)
- ...
- Chunk 8: frames 56-61 (6 frames)

Each chunk of 8 frames was still too large, causing OOM.

### After (frames_chunk_size=1):
With 62 images per GPU:
- Chunk 1: frame 0 (1 frame)
- Chunk 2: frame 1 (1 frame)
- ...
- Chunk 62: frame 61 (1 frame)

Each frame processed individually, minimizing memory footprint at the cost of longer processing time.

## Memory Trade-off

**frames_chunk_size = 1**:
- ✅ Maximum memory safety
- ❌ Slower (more chunking overhead)

**frames_chunk_size = 8** (original default):
- ✅ Faster processing
- ❌ Higher memory usage

**frames_chunk_size = 4**:
- ⚖️ Balanced approach

Users can tune based on their GPU memory and performance needs.

## Impact on 124 Image Test

For the tomato-L dataset (124 images on 2 GPUs):
- Each GPU: 62 images
- With `frames_chunk_size=1`: Each of 62 images processed separately
- Memory per chunk: ~1/62 of previous usage
- Expected result: **No more OOM!**

## Testing Strategy

1. **Verify no OOM**: Run tomato-L (124 images) with `frames_chunk_size=1`
2. **Check memory monitoring**: Confirm detailed memory logs show per-frame processing
3. **Performance baseline**: Time the run for comparison
4. **Tune upward**: Try `frames_chunk_size=2`, `4`, `8` to find optimal balance

## Files Modified

1. `repos/HY-World-2.0/hyworld2/worldrecon/pipeline.py`
   - Added parameter to `__call__`
   - Updated `_run_inference` signature
   - Added CLI argument
   - Added to call_kwargs

2. `repos/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/models/worldmirror.py`
   - Updated `forward` signature
   - Updated `_gen_all_preds_frame_sp` signature
   - Updated `_gen_all_preds` signature
   - Passed to all head calls (depth, pts, norm, gs)

## Next Steps

1. Run the 124-image test to verify OOM is resolved
2. Monitor detailed memory logs to confirm chunking behavior
3. Document optimal `frames_chunk_size` values for different GPU types
4. Consider auto-tuning based on available GPU memory
