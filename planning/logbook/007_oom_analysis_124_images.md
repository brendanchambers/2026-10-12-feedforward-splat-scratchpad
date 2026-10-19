# OOM Analysis: 124 Images (tomato-L Dataset)

**Date**: 2026-10-19
**Dataset**: tomato-L (124 images)
**Configuration**: 2x A10G GPUs (22GB each), FSDP + Sequence Parallel

## OOM Location Identified

### Exact Stack Trace
```
File: dense_head.py:354
Method: _forward_impl
Line: fused = fused.reshape(B, S, *fused.shape[1:]).float().contiguous()

Component: gs_head (Gaussian Splatting Head)
Operation: Reshape and dtype conversion
```

### Memory State at Failure
- **GPU Memory**: 22.06 GB total per GPU
- **Allocated**: 19.29 GB
- **Free**: 1.63 GB
- **Attempted allocation**: 1.93 GB
- **Result**: OOM (needed 1.93 GB but only 1.63 GB free)

### Call Stack Analysis

1. **Entry**: `pipeline.__call__` → `_run_inference`
2. **Model forward**: `WorldMirror.forward` (FSDP wrapped)
3. **Sequence parallel path**: `_gen_all_preds_frame_sp`
4. **Component**: `gs_head` (Gaussian Splatting head)
5. **Failed operation**: `dense_head._forward_impl` at reshape

### What Was Happening

The model processes 124 images in sequence parallel mode:
- Each GPU handles ~62 images (124 / 2)
- The `gs_head` processes features through multiple stages:
  1. **Extract fused features** from transformer tokens
  2. **Generate predictions** via `output_conv2`
  3. **Merge image features** with `input_merger`
  4. **Reshape back to (B, S, ...)** format ← **OOM HERE**

The reshape operation at line 354 attempts to reorganize the tensor from:
- `(B*S, C, H, W)` → `(B, S, C, H, W)`

With 124 images, this creates a massive activation tensor that exceeds available memory.

## Why Sequence Parallel Didn't Help

Sequence parallel splits images across GPUs *during* the forward pass, but:

1. **The reshape happens after merging**: By line 354, features have been processed through conv layers
2. **No collective communication yet**: The all-gather that would redistribute hasn't happened
3. **Each GPU still holds its chunk**: 62 images × feature dimensions is still too large

## Root Cause

**Memory bottleneck**: The `gs_head` (Gaussian Splatting) produces large intermediate activations.

With 124 images:
- Input shape: `(1, 124, 3, 952, 532)`
- After feature extraction: `(124, C, H, W)` where C is feature dimension
- The `.float().contiguous()` call creates a new copy in FP32
- Combined with existing allocations (19.29 GB), this pushes over 22 GB limit

## Why It Worked with 16 Images

With 16 images:
- Each GPU: ~8 images
- Activations: ~8× smaller
- Fits comfortably within 22 GB per GPU

**Critical threshold**: Somewhere between 16 and 124 images

## Sequence Parallel Flow (For Reference)

Based on the code structure:
1. Images split across GPUs (e.g., GPU0: frames 0-61, GPU1: frames 62-123)
2. Each GPU processes its chunk through transformer
3. Heads process chunks independently
4. **Collective communication** (all-gather) happens after head processing
5. Results merged on rank 0

The OOM happens at step 3, before collective communication can help.

## Solutions

### Option 1: Frame Chunking (Already Implemented!)
The `DenseHead` has a `frames_chunk_size` parameter:

```python
def forward(self, token_list, images, patch_start_idx, frames_chunk_size=None):
    if frames_chunk_size is None or frames_chunk_size >= S:
        return self._forward_impl(...)  # ← OOM here with 124 images

    # Process in chunks
    for frame_start in range(0, S, frames_chunk_size):
        # Process smaller chunks
```

**Action**: Enable `frames_chunk_size` in the pipeline call

### Option 2: Gradient Checkpointing
The code has `gradient_checkpoint` parameter but it's for training. For inference, we'd need activation checkpointing.

### Option 3: Reduce Batch Size
Process fewer images per inference run. Split 124 images into multiple batches.

### Option 4: Use More GPUs
4 GPUs would give each GPU ~31 images instead of 62.

## Next Steps

1. **Enable frame chunking** in `gs_head` forward pass
2. Find optimal `frames_chunk_size` value (test 8, 16, 32)
3. Verify memory usage with fine-grained monitoring

## Enhanced Monitoring Added

To capture this in future runs:

### 1. Added fine-grained checkpoints in `dense_head.py`:
- Extract fused features
- output_conv2 operation
- Merge image features
- **Reshape (B, S, ...) ← OOM point**

### 2. Environment variables:
```python
ENABLE_HEAD_MEMORY_MONITOR=1  # Enable head-level monitoring
ENABLE_COMM_MEMORY_MONITOR=1  # Enable collective op monitoring
```

### 3. Expected output on next run:
```
[MEM CHECKPOINT] [DenseHead/GS] ⚠️  RESHAPE B,S (B=1, S=62, shape=...) - BEFORE
  Allocated: 19.29 GB
  ...
[OOM DETECTED] [DenseHead/GS] ⚠️  RESHAPE B,S
```

This will show exactly when and where the reshape causes OOM.

## Key Insight

**The sequence parallel strategy distributes work during the forward pass, but dense head operations still process each GPU's chunk as a single large batch. With 62 images per GPU, the activation tensors in the GS head exceed memory, particularly during the reshape+float+contiguous operation.**

The solution is to chunk within each GPU's portion using the existing `frames_chunk_size` mechanism.
