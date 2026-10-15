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
):
    """
    Run the full pipeline: upload data, run inference, download results.

    Args:
        dataset_tag: Tag to organize outputs (default reads from data/incoming/{tag}/)
        images_dir: Custom path to images directory (optional)
        skip_upload: Skip the upload step (data already on Modal)
        skip_download: Skip the download step (just run inference)
    """
    print(f"🚀 Starting pipeline for dataset: {dataset_tag}")

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
    print("\n🧠 Step 2/3: Running inference...")
    inference_cmd = [
        "uv", "run", "--with", "modal", "modal", "run",
        "explore/simple_feedforward_3dgs.py",
        "--dataset-tag", dataset_tag
    ]
    result = subprocess.run(inference_cmd)
    if result.returncode != 0:
        print("❌ Inference failed")
        sys.exit(1)
    print("✅ Inference complete")

    # Step 3: Download results
    if not skip_download:
        print("\n📥 Step 3/3: Downloading results...")
        download_cmd = [
            "uv", "run", "--with", "modal", "modal", "run",
            "explore/download_results.py",
            "--dataset-tag", dataset_tag
        ]
        result = subprocess.run(download_cmd)
        if result.returncode != 0:
            print("❌ Download failed")
            sys.exit(1)
        print("✅ Download complete")
        print(f"\n🎉 Pipeline complete! Results saved to data/output/{dataset_tag}/")
    else:
        print("\n⏭️  Skipping download step")
        print(f"\n🎉 Pipeline complete!")
