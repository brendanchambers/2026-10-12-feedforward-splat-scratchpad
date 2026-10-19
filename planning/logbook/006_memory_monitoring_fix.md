# Memory Monitoring Fix - A10G GPU Compatibility

**Date**: 2026-10-19
**Issue**: PyTorch CUDA allocator config incompatibility

## Problem

Initial test run failed with:

```
[W1019 17:03:31.250815091 CUDAAllocatorConfig.h:28] Warning: expandable_segments not supported on this platform (function operator())
RuntimeError: !block->expandable_segment_ INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":2586
```

**Root Cause**: The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` setting is not supported on A10G GPUs (and potentially other GPU architectures).

## Solution

Removed `expandable_segments:True` from the CUDA allocator configuration, keeping only the universally-supported `max_split_size_mb` setting.

**Before**:
```python
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
```

**After**:
```python
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

## Files Modified

- `explore_hywm2/launch_inference.py` (both `run_inference_1gpu` and `run_inference_fsdp`)
- `planning/oom-debugging-guide.md` (updated documentation)

## Impact

- Memory monitoring still works correctly
- `max_split_size_mb:512` still helps with memory fragmentation
- Compatible with A10G, A100, and other GPU types
- Memory checkpoints remain fully functional

## Next Test

The memory monitoring instrumentation should now work properly. When you run inference with more images than fit in memory, you'll see output like:

```
================================================================================
[MEM CHECKPOINT] Model forward pass (main OOM location) - BEFORE
================================================================================
  Allocated:        8.732 GB
  Reserved:         9.500 GB
  Peak so far:      8.732 GB
================================================================================

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[OOM DETECTED] Model forward pass (main OOM location)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Memory state when OOM occurred:
  Allocated before:    8.732 GB
  ...
```

This will pinpoint exactly where the OOM occurs during inference.
