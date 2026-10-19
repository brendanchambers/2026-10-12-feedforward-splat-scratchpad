# frames_chunk_size Investigation

**Date**: 2026-10-19
**Question**: What is the current value of `frames_chunk_size` during inference?

## Answer: Default Value is 8, BUT It's Not Being Used

### Code Analysis

**1. DenseHead Forward Signature** (`dense_head.py:238-243`):
```python
def forward(
    self,
    token_list: List[torch.Tensor],
    images: torch.Tensor,
    patch_start_idx: int,
    frames_chunk_size: int = 8,  # ← DEFAULT VALUE
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

**Default value**: `frames_chunk_size = 8`

**2. Logic** (`dense_head.py:262-263`):
```python
# Process all frames together if chunk size not specified or large enough
if frames_chunk_size is None or frames_chunk_size >= S:
    return self._forward_impl(token_list, images, patch_start_idx)  # ← No chunking!
```

**3. Actual Call Site** (`worldmirror.py:476, 484`):
```python
# Line 476 (with depth mask)
gs_feat_p, gs_depth_p, gs_depth_conf_p, gs_dmask_p = self.gs_head(
    gs_token_chunk, images=gs_imgs_chunk, patch_start_idx=patch_start_idx,
)

# Line 484 (without depth mask)
gs_feat_p, gs_depth_p, gs_depth_conf_p = self.gs_head(
    gs_token_chunk, images=gs_imgs_chunk, patch_start_idx=patch_start_idx,
)
```

**`frames_chunk_size` is NOT passed** → Uses default value of **8**

## The Problem

With 124 images on 2 GPUs in sequence parallel mode:

1. **Sequence parallel splits**: Each GPU gets ~62 images
2. **gs_head receives**: `gs_imgs_chunk` with shape `(B=1, S=62, C, H, W)`
3. **Chunking check**: `if frames_chunk_size >= S:` → `if 8 >= 62:` → **False**
4. **Should chunk**: According to the logic, it SHOULD process in chunks of 8

## Wait... Why Did It Still OOM?

Let me re-examine the logic more carefully:

**Line 262**: `if frames_chunk_size is None or frames_chunk_size >= S:`

This means:
- If `frames_chunk_size >= S` → process all at once (no chunking)
- If `frames_chunk_size < S` → process in chunks

With S=62 and frames_chunk_size=8:
- `8 >= 62` → **False**
- Should go to line 265+ and process in chunks!

## Why Did the OOM Still Happen Then?

Let me check if there are other heads being called...

Looking at the stack trace from the error:
```
File "/root/repo/repos/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/models/worldmirror.py", line 476
    gs_feat_p, gs_depth_p, gs_depth_conf_p, gs_dmask_p = self.gs_head(...)

File "dense_head.py", line 266, in forward
    gs, preds, conf, depth_mask = self._forward_impl(...)
```

**Wait!** The stack trace shows it's calling `_forward_impl` directly at line 266!

Let me check what's at line 266 in dense_head.py...

Actually, looking at our instrumented code now, the line numbers have shifted. But the original error showed:
```
File "dense_head.py", line 266, in forward
    gs, preds, conf, depth_mask = self._forward_impl(...)
```

This is INSIDE the chunking loop! Let me look at that code again.

## The Real Answer

Looking at the chunking code (lines 261-310):

```python
for frame_start in range(0, S, frames_chunk_size):
    frame_end = min(frame_start + frames_chunk_size, S)

    if self.is_gsdpt:
        if self.enable_depth_mask:
            gs, preds, conf, depth_mask = self._forward_impl(
                token_list, images, patch_start_idx, frame_start, frame_end
            )
```

**The chunking IS working!**

Each call to `_forward_impl` receives `frame_start` and `frame_end`, and at line 334:
```python
if frame_start is not None and frame_end is not None:
    images = images[:, frame_start:frame_end].contiguous()
```

So each chunk processes ~8 frames.

## So Why the OOM?

The OOM happened at line 354:
```python
fused = fused.reshape(B, S, *fused.shape[1:]).float().contiguous()
```

Where `S` is the **chunk size** (8), not the full 62!

**The issue**: Even processing 8 frames at a time, the reshape operation still requires too much memory!

## Calculation

With chunk_size=8 on one GPU:
- Input: `(B=1, S=8, C=3, H=952, W=532)`
- After processing through layers, `fused` might be: `(B*S=8, C_feat, H_feat, W_feat)`
- Reshape to: `(B=1, S=8, C_feat, H_feat, W_feat)`

If C_feat is large (e.g., 256 channels) and H_feat, W_feat are still high resolution, this tensor can be huge:
- `1 × 8 × 256 × H × W × 4 bytes (float32)`

## Conclusion

**Current value**: `frames_chunk_size = 8` (default)

**Is it being used?**: Yes, chunking IS active for S > 8

**Problem**: Even with chunking at 8 frames, **each chunk still OOMs** due to large intermediate activations in the dense head.

**Solution needed**:
1. Reduce `frames_chunk_size` to something smaller (4, 2, or even 1)
2. Or optimize the reshape operation to avoid the memory spike
3. Or use activation checkpointing within the head

The chunking helps, but with 62 images per GPU and high-resolution features, even 8-frame chunks exceed memory limits.
