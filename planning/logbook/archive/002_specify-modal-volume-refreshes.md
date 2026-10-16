# Modal Volume Refresh Implementation

## Problem Statement

Data syncing issues between sequential Modal jobs due to stale volume state. Modal volumes don't automatically refresh to show changes made by other jobs after container startup.

## Background: Modal Volume Best Practices

### Key Concepts

1. **Volume Reload Pattern**: "Writers commit, readers reload"
   - Writers call `volume.commit()` after writing data
   - Readers call `volume.reload()` before reading data from other jobs

2. **Auto-reload timing**: Volumes auto-reload only at container startup
   - Subsequent writes by other jobs won't be visible until manual `reload()`
   - Cannot reload while files are open ("volume busy" error)

3. **Data persistence**: Volume data persists across runs
   - Must be explicitly deleted via `modal volume delete` or programmatically
   - Old data from previous runs can cause issues

## Implementation

### 1. Added `volume.reload()` Calls

**Location: `simple_feedforward_3dgs.py:81`**
```python
def run_inference(...):
    # Reload volume to see latest uploaded data
    data_volume.reload()
    print(f"Starting inference for dataset '{dataset_tag}'")
```

**Location: `optimize_3dgs.py:91`**
```python
def run_optimization(...):
    # Reload volume to see latest inference data
    data_volume.reload()
    print(f"Starting optimization for dataset '{dataset_tag}'")
```

**Location: `download_results.py:16`**
```python
def list_and_read_files(...):
    # Reload volume to see latest optimization/inference data
    data_volume.reload()
    remote_path = Path("/data/output") / dataset_tag
```

### 2. Data Deletion Functionality

**New file: `delete_data.py`**
- Standalone script for manual data cleanup
- Deletes both `/data/incoming/{tag}` and `/data/output/{tag}`
- Usage: `uv run --with modal modal run explore/delete_data.py --dataset-tag my-images`

**Modified: `pipeline.py`**
- Added `delete_dataset_data()` function
- Added `--delete-data` flag to main entrypoint
- Runs as "Step 0" before upload when flag is set

```python
@app.function(volumes={"/data": volume})
def delete_dataset_data(dataset_tag: str):
    """Delete incoming and output data for a dataset from the volume."""
    paths_to_delete = [
        Path("/data/incoming") / dataset_tag,
        Path("/data/output") / dataset_tag,
    ]
    # ... deletion logic ...
    volume.commit()
```

### 3. Documentation Updates

**Modified: `README.md`**
- Added "Data Management" section
- Documented `--delete-data` flag usage
- Explained volume synchronization behavior
- Provided examples for clearing stale data

## Data Flow with Reload Pattern

```
Upload (commit) → Inference (reload → commit) → Optimize (reload → commit) → Download (reload)
```

Each stage:
1. Reloads volume to see fresh data from previous stage
2. Performs its work
3. Commits changes for next stage

## Usage Examples

### Fresh Run with Data Cleanup
```bash
uv run --with modal modal run explore/pipeline.py \
  --dataset-tag tomato-small \
  --delete-data
```

### Manual Data Cleanup
```bash
uv run --with modal modal run explore/delete_data.py \
  --dataset-tag tomato-small
```

### Individual Steps (with reload)
```bash
# Upload (no reload needed - writes fresh data)
uv run --with modal modal run explore/upload_data.py --dataset-tag my-images

# Inference (reloads to see uploaded images)
uv run --with modal modal run explore/simple_feedforward_3dgs.py --dataset-tag my-images

# Optimization (reloads to see inference outputs)
uv run --with modal modal run explore/optimize_3dgs.py --dataset-tag my-images

# Download (reloads to see optimization outputs)
uv run --with modal modal run explore/download_results.py --dataset-tag my-images
```

## Files Modified

1. `explore/simple_feedforward_3dgs.py` - Added reload before inference
2. `explore/optimize_3dgs.py` - Added reload before optimization
3. `explore/download_results.py` - Added reload before download
4. `explore/pipeline.py` - Added delete function and --delete-data flag
5. `explore/delete_data.py` - New standalone deletion script
6. `explore/README.md` - Added data management documentation

## Benefits

1. **Prevents stale data issues** - Each job sees latest committed data
2. **Fresh runs** - Easy cleanup with `--delete-data` flag
3. **Follows best practices** - Implements Modal's "writers commit, readers reload" pattern
4. **Better debugging** - Clear data flow and explicit sync points
5. **Flexible cleanup** - Both integrated (pipeline) and standalone (delete_data.py) options

## Notes

- Reload must be called before opening any files (avoid "volume busy" errors)
- Upload step doesn't need reload (writes fresh data, no dependencies)
- All commits are already in place from previous implementation
- Delete operation removes entire directory trees for dataset tag
