# OOM Debugging Guide for FSDP + Sequence Parallel Inference

## Overview

This guide documents the memory monitoring instrumentation added to help identify the exact location of out-of-memory (OOM) errors during FSDP + sequence parallel inference.

**Date**: 2026-10-19
**Purpose**: Identify OOM trigger points when scaling up the number of input images

## What Was Added

### 1. Memory Monitoring Utility (`repos/HY-World-2.0/hyworld2/worldrecon/memory_monitor.py`)

A context manager that tracks GPU memory before/after operations:

```python
from memory_monitor import memory_checkpoint

with memory_checkpoint("Operation name", device):
    # Your code here
    output = model(input)
```

**Features**:
- Tracks allocated, reserved, and peak memory
- Automatically catches OOM errors with detailed debugging info
- Shows memory deltas for each operation
- Minimal overhead when enabled

### 2. Pipeline Instrumentation

Added memory checkpoints at critical locations in `pipeline.py`:

1. **Image loading and preprocessing** (`_run_inference:629`)
   - Monitors memory during image tensor creation and GPU transfer

2. **Prior camera/depth loading** (`_run_inference:651`)
   - Tracks memory if using prior camera poses or depth maps

3. **Model forward pass** (`_run_inference:673`)
   - **Most critical checkpoint** - monitors the main inference operation
   - This is where OOMs are most likely to occur with large batch sizes

### 3. Collective Communication Monitoring

Instrumented `all_gather` and `all2all` operations in `hyworldmirror/comm/communication.py`:

- Controlled by environment variable `ENABLE_COMM_MEMORY_MONITOR=1`
- Shows memory usage for each collective operation
- Helps identify if OOM occurs during sequence parallel gather/scatter

### 4. PyTorch CUDA Debugging

Added environment variables in `launch_inference.py`:

```python
env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
env["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # FSDP mode only
env["ENABLE_COMM_MEMORY_MONITOR"] = "1"
```

**Note**: `expandable_segments:True` is not supported on all GPU platforms (e.g., A10G GPUs). We use only `max_split_size_mb` for compatibility.

## How to Use

### Running with Memory Monitoring

The monitoring is **always enabled** by default. Just run inference normally:

```bash
# Single GPU
modal run explore_hywm2/launch_inference.py --dataset-tag your-dataset --gpu-mode 1gpu

# Multi-GPU with FSDP
modal run explore_hywm2/launch_inference.py --dataset-tag your-dataset --gpu-mode fsdp
```

### Reading the Output

When an operation completes successfully:

```
================================================================================
[MEM CHECKPOINT] Image loading and preprocessing - BEFORE
================================================================================
  Allocated:        2.450 GB
  Reserved:         3.120 GB
  Peak so far:      2.450 GB
================================================================================

[MEM CHECKPOINT] Image loading and preprocessing - AFTER
================================================================================
  Allocated:        8.732 GB  (Δ  +6.282 GB)
  Reserved:         9.500 GB  (Δ  +6.380 GB)
  Peak:             8.732 GB  (Δ  +6.282 GB)
================================================================================
```

When an OOM occurs:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[OOM DETECTED] Model forward pass (main OOM location)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Memory state when OOM occurred:
  Allocated before:    8.732 GB
  Reserved before:     9.500 GB
  Peak before:         8.732 GB

This operation attempted to allocate more memory than available.

Full error message:
  CUDA out of memory. Tried to allocate 12.50 GiB...

Stack trace:
  [Full Python traceback here]
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Scaling Strategy

To find the exact breaking point:

1. **Start small**: Begin with a number of images you know works (e.g., 2-4)
2. **Binary search**: Double until OOM, then bisect
3. **Analyze**: Look at the checkpoint that failed to understand which operation caused OOM

Example progression:
- 2 images ✓ (baseline)
- 4 images ✓
- 8 images ✓
- 16 images ✗ OOM at "Model forward pass"
- 12 images ? (test to narrow down)

## What the Output Tells You

### If OOM occurs at "Image loading and preprocessing"
- Input images are too large or too many
- Consider reducing target resolution or batch size before model

### If OOM occurs at "Model forward pass"
- Model activations exceed GPU memory
- In FSDP mode: check if sequence parallelism is properly distributing work
- Possible solutions:
  - Reduce number of images per batch
  - Enable activation checkpointing (if available)
  - Use CPU offloading (`--fsdp_cpu_offload`)

### If OOM occurs at "all_gather" or "all2all" (collective ops)
- Sequence parallel gather operation is creating too large a tensor
- This happens when gathering results from all GPUs
- The gathered tensor is `num_gpus × local_size`
- Solutions:
  - Use more GPUs to reduce per-GPU memory
  - Gather in chunks rather than all at once

## Disabling Memory Monitoring

If you want to disable the collective communication monitoring (for performance):

In `launch_inference.py`, comment out or remove:
```python
env["ENABLE_COMM_MEMORY_MONITOR"] = "1"
```

The pipeline-level checkpoints cannot be disabled without code changes, but they have minimal overhead.

## Technical Details

### Memory Checkpoint Implementation

The `memory_checkpoint` context manager:
1. Calls `torch.cuda.synchronize()` to ensure all operations are complete
2. Records memory stats via `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`
3. Executes the wrapped code
4. On success: logs memory deltas
5. On OOM: catches the exception, logs detailed state, and re-raises

### FSDP Considerations

In FSDP mode:
- Each rank has its own memory checkpoint output
- Look for the rank that OOMs first (may not be rank 0)
- Memory usage should be similar across ranks in sequence parallel mode

### Sequence Parallel Workflow

The typical memory pattern in sequence parallel:
1. Images loaded: split across GPUs (low memory per GPU)
2. Model forward: each GPU processes its chunk (moderate memory)
3. All-gather: collect results from all GPUs (**memory spike**)
4. Post-processing: full results on rank 0

OOM most commonly occurs at step 3 (all-gather).

## Next Steps

If you identify an OOM location and want to optimize:
1. Check the operation's input/output sizes
2. Verify sequence parallel is distributing work correctly
3. Consider gradient checkpointing for that operation
4. Profile to see if memory can be released earlier

## Files Modified

- `repos/HY-World-2.0/hyworld2/worldrecon/memory_monitor.py` (new)
- `repos/HY-World-2.0/hyworld2/worldrecon/pipeline.py` (instrumented)
- `repos/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/comm/communication.py` (instrumented)
- `explore_hywm2/launch_inference.py` (environment variables added)
