"""
Simple pipeline to run the full 3DGS workflow: upload -> inference -> download
"""
import modal
import subprocess
import sys

app = modal.App("simple-feedforward-3dgs-pipeline")
volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(volumes={"/data": volume})
def delete_dataset_data(dataset_tag: str, model_tag: str = "hy-wm-1.1"):
    """Delete incoming and output data for a dataset from the volume."""
    import shutil
    from pathlib import Path

    paths_to_delete = [
        Path("/data/incoming") / dataset_tag,
        Path("/data/output") / model_tag / dataset_tag,
    ]

    deleted = []
    for path in paths_to_delete:
        if path.exists():
            print(f"🗑️  Deleting {path}...")
            shutil.rmtree(path)
            deleted.append(str(path))
        else:
            print(f"⏭️  Path does not exist: {path}")

    if deleted:
        volume.commit()
        print(f"✅ Deleted {len(deleted)} path(s) from volume")
    else:
        print("ℹ️  No data to delete")

    return deleted


@app.local_entrypoint()
def main(
    dataset_tag: str,
    model_tag: str = "hy-wm-1.1",
    images_dir: str | None = None,
    skip_upload: bool = False,
    skip_download: bool = False,
    skip_optimize: bool = False,
    optimize_steps: int = 3000,
    verbose: bool = False,
    delete_data: bool = False,
):
    """
    Run the full pipeline: upload data, run feedforard inference, post hoc classical optimize (optional), download results.

    Args:
        dataset_tag: Tag to organize outputs (default reads from data/incoming/{tag}/)
        model_tag: Model version to use (e.g., hy-wm-1.1, hy-wm-2.0)
        images_dir: Custom path to images directory (optional)
        skip_upload: Skip the upload step (data already on Modal)
        skip_download: Skip the download step (just run inference)
        skip_optimize: Skip the post-3DGS optimization step
        optimize_steps: Number of optimization steps (default: 3000)
        verbose: Enable verbose output during inference
        delete_data: Delete existing dataset data from volume before starting
    """
    # Step 0: Delete existing data if requested
    if delete_data:
        print(f"\n🗑️  Step 0: Deleting existing data for '{dataset_tag}' (model: {model_tag})...")
        delete_dataset_data.remote(dataset_tag, model_tag)
        print("✅ Data deletion complete")

    # Map model tag to inference script
    # Convert model tag (e.g., hy-wm-1.1) to script name (e.g., inference_hy-wm-1p1.py)
    # Replace dots with 'p' and convert to underscore format for some versions
    model_to_script = {
        "hy-wm-2.0": "launch_inference.py",
    }

    inference_script = model_to_script.get(model_tag)
    if not inference_script:
        print(f"❌ Unknown model tag: {model_tag}")
        print(f"   Available models: {', '.join(model_to_script.keys())}")
        sys.exit(1)

    print(f"🚀 Starting pipeline for dataset: {dataset_tag} with model: {model_tag}")

    # Step 1: Upload data
    if not skip_upload:
        print("\n📤 Step 1/3: Uploading data...")
        upload_cmd = [
            "uv", "run", "--with", "modal", "modal", "run",
            "explore/upload_data.py",
            "--dataset-tag", dataset_tag
        ]
        if images_dir:
            upload_cmd.extend(["--images-dir", images_dir])

        result = subprocess.run(upload_cmd)
        if result.returncode != 0:
            print("❌ Upload failed")
            sys.exit(1)
        print("✅ Upload complete")
    else:
        print("\n⏭️  Skipping upload step")

    # Step 2: Run inference
    print(f"\n🧠 Step 2/3: Running inference with {model_tag}...")
    inference_cmd = [
        "uv", "run", "--with", "modal", "modal", "run",
        f"explore/{inference_script}",
        "--dataset-tag", dataset_tag,
        "--model-tag", model_tag
    ]
    if verbose:
        inference_cmd.append("--verbose")
    result = subprocess.run(inference_cmd)
    if result.returncode != 0:
        print("❌ Inference failed")
        sys.exit(1)
    print("✅ Inference complete")

    # Step 3: Run post-3DGS optimization (optional)
    # if not skip_optimize:
    #     print(f"\n🔧 Step 3/4: Running post-3DGS optimization ({optimize_steps} steps)...")
    #     optimize_cmd = [
    #         "uv", "run", "--with", "modal", "modal", "run",
    #         "explore/optimize_3dgs.py",
    #         "--dataset-tag", dataset_tag,
    #         "--max-steps", str(optimize_steps)
    #     ]
    #     result = subprocess.run(optimize_cmd)
    #     if result.returncode != 0:
    #         print("❌ Optimization failed")
    #         sys.exit(1)
    #     print("✅ Optimization complete")
    # else:
    #     print("\n⏭️  Skipping optimization step")

    # Step 4: Download results
    if not skip_download:
        print("\n📥 Step 4/4: Downloading results...")
        download_cmd = [
            "uv", "run", "--with", "modal", "modal", "run",
            "explore/download_results.py",
            "--dataset-tag", dataset_tag,
            "--model-tag", model_tag
        ]
        result = subprocess.run(download_cmd)
        if result.returncode != 0:
            print("❌ Download failed")
            sys.exit(1)
        print("✅ Download complete")
        print(f"\n🎉 Pipeline complete! Results saved to data/output/{model_tag}/{dataset_tag}/")
        if not skip_optimize:
            print(f"   - Original 3DGS: data/output/{model_tag}/{dataset_tag}/gaussians.ply")
            print(f"   - Optimized 3DGS: data/output/{model_tag}/{dataset_tag}/optimized/")
    else:
        print("\n⏭️  Skipping download step")
        print(f"\n🎉 Pipeline complete!")
