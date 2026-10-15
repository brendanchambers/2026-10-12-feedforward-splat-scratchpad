"""
Simple pipeline to run the full 3DGS workflow: upload -> inference -> download
"""
import modal
import subprocess
import sys

app = modal.App("simple-feedforward-3dgs-pipeline")


@app.local_entrypoint()
def main(
    dataset_tag: str,
    images_dir: str | None = None,
    skip_upload: bool = False,
    skip_download: bool = False,
    skip_optimize: bool = False,
    optimize_steps: int = 3000,
    verbose: bool = False,
):
    """
    Run the full pipeline: upload data, run feedforard inference, post hoc classical optimize (optional), download results.

    Args:
        dataset_tag: Tag to organize outputs (default reads from data/incoming/{tag}/)
        images_dir: Custom path to images directory (optional)
        skip_upload: Skip the upload step (data already on Modal)
        skip_download: Skip the download step (just run inference)
        skip_optimize: Skip the post-3DGS optimization step
        optimize_steps: Number of optimization steps (default: 3000)
        verbose: Enable verbose output during inference
    """
    print(f"🚀 Starting pipeline for dataset: {dataset_tag}")

    # Step 1: Upload data
    if not skip_upload:
        print("\n📤 Step 1/3: Uploading data...")
        upload_cmd = [
            "uv", "run", "--with", "modal", "modal", "run",
            "explore/sync_data_up.py",
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
    print("\n🧠 Step 2/3: Running inference...")
    inference_cmd = [
        "uv", "run", "--with", "modal", "modal", "run",
        "explore/simple_feedforward_3dgs.py",
        "--dataset-tag", dataset_tag
    ]
    if verbose:
        inference_cmd.append("--verbose")
    result = subprocess.run(inference_cmd)
    if result.returncode != 0:
        print("❌ Inference failed")
        sys.exit(1)
    print("✅ Inference complete")

    # Step 3: Run post-3DGS optimization (optional)
    if not skip_optimize:
        print(f"\n🔧 Step 3/4: Running post-3DGS optimization ({optimize_steps} steps)...")
        optimize_cmd = [
            "uv", "run", "--with", "modal", "modal", "run",
            "explore/optimize_3dgs.py",
            "--dataset-tag", dataset_tag,
            "--max-steps", str(optimize_steps)
        ]
        result = subprocess.run(optimize_cmd)
        if result.returncode != 0:
            print("❌ Optimization failed")
            sys.exit(1)
        print("✅ Optimization complete")
    else:
        print("\n⏭️  Skipping optimization step")

    # Step 4: Download results
    if not skip_download:
        print("\n📥 Step 4/4: Downloading results...")
        download_cmd = [
            "uv", "run", "--with", "modal", "modal", "run",
            "explore/sync_data_down.py",
            "--dataset-tag", dataset_tag
        ]
        result = subprocess.run(download_cmd)
        if result.returncode != 0:
            print("❌ Download failed")
            sys.exit(1)
        print("✅ Download complete")
        print(f"\n🎉 Pipeline complete! Results saved to data/output/{dataset_tag}/")
        if not skip_optimize:
            print(f"   - Original 3DGS: data/output/{dataset_tag}/gaussians.ply")
            print(f"   - Optimized 3DGS: data/output/{dataset_tag}/optimized/")
    else:
        print("\n⏭️  Skipping download step")
        print(f"\n🎉 Pipeline complete!")
