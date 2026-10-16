"""
Upload input images to Modal volume for inference.
"""

from pathlib import Path
import modal

app = modal.App("feedforward-3dgs-upload")
volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(volumes={"/data": volume})
def upload_files(files_data: list[tuple[bytes, str]]):
    """Upload files to volume."""
    from pathlib import Path

    for file_bytes, remote_path in files_data:
        full_path = Path("/data") / remote_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(file_bytes)

    volume.commit()


@app.local_entrypoint()
def main(dataset_tag: str = "tomato-bed-tiny", images_dir: str = None):
    """Upload images to Modal volume."""
    from pathlib import Path

    # Use images_dir if provided, otherwise default to data/incoming/{dataset_tag}
    local_dir = images_dir if images_dir else f"data/incoming/{dataset_tag}"
    local_path = Path(local_dir)
    if not local_path.exists():
        raise ValueError(f"Local directory does not exist: {local_path}")

    print(f"📦 Uploading dataset '{dataset_tag}' from {local_dir} to Modal volume...")

    # Collect all image files
    image_files = []
    image_exts = [".jpg", ".jpeg", ".png"]
    for ext in image_exts:
        image_files.extend(local_path.glob(f"*{ext}"))

    if not image_files:
        raise ValueError(f"No image files found in {local_path}")

    print(f"📸 Found {len(image_files)} images")

    # Prepare files for upload
    remote_base = f"incoming/{dataset_tag}"
    files_data = [(img.read_bytes(), f"{remote_base}/{img.name}") for img in image_files]

    upload_files.remote(files_data)

    print(f"✅ Uploaded {len(image_files)} files to /{remote_base} in Modal volume")
