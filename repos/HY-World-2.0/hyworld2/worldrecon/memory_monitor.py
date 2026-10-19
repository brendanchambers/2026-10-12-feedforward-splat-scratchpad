"""
Memory monitoring utilities for debugging OOM errors in FSDP + sequence parallel inference.

Usage:
    from memory_monitor import memory_checkpoint

    with memory_checkpoint("Forward pass", device):
        output = model(input)
"""

import torch
import traceback
from contextlib import contextmanager


@contextmanager
def memory_checkpoint(label: str, device, threshold_gb=None, verbose=False):
    """Monitor GPU memory usage around a code block and detect OOMs.

    Args:
        label: Description of the operation being monitored
        device: torch.device to monitor
        threshold_gb: Optional warning threshold in GB
        verbose: If True, print detailed memory stats

    Yields:
        None

    Example:
        with memory_checkpoint("Model forward", torch.device("cuda:0")):
            output = model(input)
    """
    if device.type != "cuda":
        yield
        return

    # Capture state before operation
    torch.cuda.synchronize(device)
    before_allocated = torch.cuda.memory_allocated(device) / 1024**3
    before_reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak_before = torch.cuda.max_memory_allocated(device) / 1024**3

    if verbose:
        print(f"\n{'='*80}")
        print(f"[MEM CHECKPOINT] {label} - BEFORE")
        print(f"{'='*80}")
        print(f"  Allocated:    {before_allocated:8.3f} GB")
        print(f"  Reserved:     {before_reserved:8.3f} GB")
        print(f"  Peak so far:  {peak_before:8.3f} GB")
        print(f"{'='*80}")

    try:
        yield

        # Capture state after successful operation
        torch.cuda.synchronize(device)
        after_allocated = torch.cuda.memory_allocated(device) / 1024**3
        after_reserved = torch.cuda.memory_reserved(device) / 1024**3
        peak_after = torch.cuda.max_memory_allocated(device) / 1024**3

        delta_allocated = after_allocated - before_allocated
        delta_reserved = after_reserved - before_reserved
        peak_delta = peak_after - peak_before

        if verbose:
            print(f"\n[MEM CHECKPOINT] {label} - AFTER")
            print(f"{'='*80}")
            print(f"  Allocated:    {after_allocated:8.3f} GB  (Δ {delta_allocated:+8.3f} GB)")
            print(f"  Reserved:     {after_reserved:8.3f} GB  (Δ {delta_reserved:+8.3f} GB)")
            print(f"  Peak:         {peak_after:8.3f} GB  (Δ {peak_delta:+8.3f} GB)")
            print(f"{'='*80}\n")

            # Warn if threshold exceeded
            if threshold_gb and peak_after > threshold_gb:
                print(f"⚠️  WARNING: Peak memory ({peak_after:.2f} GB) exceeds threshold ({threshold_gb} GB)")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # OOM detected - provide detailed debugging info
            torch.cuda.synchronize(device)

            print(f"\n{'!'*80}")
            print(f"[OOM DETECTED] {label}")
            print(f"{'!'*80}")
            print(f"Memory state when OOM occurred:")
            print(f"  Allocated before: {before_allocated:8.3f} GB")
            print(f"  Reserved before:  {before_reserved:8.3f} GB")
            print(f"  Peak before:      {peak_before:8.3f} GB")
            print(f"\nThis operation attempted to allocate more memory than available.")
            print(f"\nFull error message:")
            print(f"  {str(e)}")
            print(f"\nStack trace:")
            traceback.print_exc()
            print(f"{'!'*80}\n")

            # Try to get GPU info
            try:
                import pynvml
                pynvml.nvmlInit()
                device_idx = device.index if device.index is not None else 0
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = memory_info.total / 1024**3
                print(f"GPU {device_idx} total memory: {total_gb:.2f} GB")
                pynvml.nvmlShutdown()
            except Exception:
                pass

        # Re-raise the exception
        raise


def log_memory_summary(device, label="Memory Summary"):
    """Log a summary of current GPU memory state.

    Args:
        device: torch.device to monitor
        label: Label for this memory summary
    """
    if device.type != "cuda":
        return

    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3

    print(f"\n[{label}]")
    print(f"  Allocated: {allocated:.3f} GB")
    print(f"  Reserved:  {reserved:.3f} GB")
    print(f"  Peak:      {peak:.3f} GB\n")


def reset_peak_memory(device):
    """Reset peak memory stats for cleaner per-operation tracking.

    Args:
        device: torch.device to reset
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
