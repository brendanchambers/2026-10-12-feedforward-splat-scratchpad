# Data Sync Strategy Analysis

## Current Approach: Modal Volumes

**Implementation:**
- Upload: `upload_data.py` sends files to Modal Volume
- Inference: `simple_feedforward_3dgs.py` reads/writes from Volume
- Download: `download_results.py` retrieves results from Volume

**Pros:**
- Simple, filesystem-like interface
- Multi-region distribution (good for Modal's compute pool)
- Up to 2.5 GB/s bandwidth
- Works well for current use case (<50 images per dataset, well under 50k file limit)
- No external dependencies

**Cons:**
- Requires explicit upload/download steps (3 separate commands)
- Slower iteration cycle
- Limited to Modal ecosystem
- Commit latency scales with file count

## Alternative 1: Cloud Bucket Mounts (S3/R2/GCS)

**Best for:** Production workflows, sharing data across systems

**Pros:**
- Direct access to external storage
- No explicit upload/download steps needed
- Data persists outside Modal
- Can integrate with other tools/pipelines
- Good for large-scale datasets

**Cons:**
- More complex setup (credentials, bucket management)
- Append/seek limitations (AWS Mountpoint restrictions)
- Requires cloud storage account
- Potential egress costs (except Cloudflare R2)

**Example pattern:**
```python
from modal import CloudBucketMount

bucket = CloudBucketMount(
    "my-bucket",
    key_prefix="3dgs-datasets/",
)

@app.function(cloud_bucket_mounts={"/data": bucket})
def run_inference(dataset_tag: str):
    # Read directly from /data/incoming/{dataset_tag}
    # Write directly to /data/output/{dataset_tag}
```

## Alternative 2: Direct Parameter Passing

**Best for:** Small datasets (few images)

Pass image data directly as function parameters instead of using persistent storage.

**Cons:**
- Not practical for 16 images at 518px resolution
- Network overhead for each invocation
- Not viable for current use case

## Alternative 3: Volume v2 (Beta)

**Best for:** Large-scale workflows with many files

Same interface as current approach but supports unlimited files and better concurrency.

**Upgrade:**
```python
volume = modal.Volume.from_name(
    "feedforward-3dgs-data",
    create_if_missing=True,
    version=2
)
```

## Recommendations

### Path A: Keep Volumes (simplest)
Current approach is solid for exploration/prototyping:
- Dataset sizes are small (<50 images)
- Workflow is clear and predictable
- No external dependencies

**Minor optimization:** Upgrade to Volume v2 for better performance.

### Path B: Switch to Cloud Buckets (more scalable)
Consider if planning to:
- Run many experiments with different datasets
- Share data with other tools/collaborators
- Build a production pipeline
- Avoid vendor lock-in

**Recommendation:** Cloudflare R2 has zero egress fees, making it cheaper for downloads.

## Decision Criteria

- **Prototyping/exploration:** Stick with Modal Volumes
- **Production pipeline:** Switch to Cloud Buckets (R2 preferred)
- **Many files (>50k):** Upgrade to Volume v2
- **Data sharing needs:** Cloud Buckets required

## References

- [Modal Volumes Documentation](https://modal.com/docs/guide/volumes)
- [Modal Cloud Bucket Mounts](https://modal.com/docs/guide/cloud-bucket-mounts)
- [Modal Network File Systems (Deprecated)](https://modal.com/docs/guide/network-file-systems)
