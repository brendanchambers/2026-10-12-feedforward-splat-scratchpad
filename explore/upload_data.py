"""
Upload input images to Modal volume for inference.
"""

from pathlib import Path
import modal

app = modal.App("feedforward-3dgs-upload")


@app.local_entrypoint()
def main(local_dir: str = "data/incoming/tomato-bed-tiny", remote_dir: str = "tomato-bed-tiny"):
    """Upload images to Modal volume."""
    from pathlib import Path

    local_path = Path(local_dir)
    if not local_path.exists():
        raise ValueError(f"Local directory does not exist: {local_path}")

    print(f"📦 Uploading from {local_dir} to Modal volume...")

    # Create or get volume
    volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)

    # Collect all image files
    image_files = []
    image_exts = [".jpg", ".jpeg", ".png"]
    for ext in image_exts:
        image_files.extend(local_path.glob(f"*{ext}"))

    if not image_files:
        raise ValueError(f"No image files found in {local_path}")

    print(f"📸 Found {len(image_files)} images")

    # Upload files
    remote_base = f"incoming/{remote_dir}"
    files_to_upload = [(str(img), f"{remote_base}/{img.name}") for img in image_files]
    volume.batch_upload(files_to_upload)

    print(f"✅ Uploaded {len(image_files)} files to /{remote_base} in Modal volume")
