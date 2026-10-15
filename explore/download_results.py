"""
Download inference results from Modal volume to local directory.
"""

from pathlib import Path
import modal

app = modal.App("feedforward-3dgs-download")
data_volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(volumes={"/data": data_volume})
def list_and_read_files(dataset_tag: str):
    """List files in volume and return their contents."""
    from pathlib import Path

    remote_path = Path("/data/output") / dataset_tag

    if not remote_path.exists():
        raise ValueError(f"Remote directory does not exist: {remote_path}")

    files_data = []
    for file in remote_path.rglob("*"):
        if file.is_file():
            rel_path = str(file.relative_to(remote_path))
            content = file.read_bytes()
            files_data.append((rel_path, content))
            print(f"📥 Reading: {rel_path}")

    return files_data


@app.local_entrypoint()
def main(dataset_tag: str = "tomato-bed-tiny"):
    """Download results from Modal volume."""
    from pathlib import Path

    local_dir = f"data/output/{dataset_tag}"
    print(f"📦 Downloading dataset '{dataset_tag}' to {local_dir}...")

    # Get files from Modal
    files_data = list_and_read_files.remote(dataset_tag)

    # Write locally
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    for rel_path, content in files_data:
        dest = local_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        print(f"💾 Saved: {rel_path}")

    print(f"✅ Downloaded {len(files_data)} files to {local_path}")
