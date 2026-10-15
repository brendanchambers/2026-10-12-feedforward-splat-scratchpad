"""
Download inference results from Modal volume to local directory.
"""

from pathlib import Path
import modal

app = modal.App("feedforward-3dgs-download")
data_volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(volumes={"/data": data_volume})
def download_results(remote_dir: str, local_dir: str):
    """Download results from Modal volume to local directory."""
    import shutil
    from pathlib import Path

    remote_path = Path("/data/output") / remote_dir
    local_path = Path(local_dir)

    if not remote_path.exists():
        raise ValueError(f"Remote directory does not exist: {remote_path}")

    # Create local directory
    local_path.mkdir(parents=True, exist_ok=True)

    # Copy all files
    copied = 0
    for file in remote_path.rglob("*"):
        if file.is_file():
            rel_path = file.relative_to(remote_path)
            dest = local_path / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest)
            copied += 1
            print(f"📥 Downloaded: {rel_path}")

    print(f"✅ Downloaded {copied} files to {local_path}")


@app.local_entrypoint()
def main(remote_dir: str = "tomato-bed-tiny", local_dir: str = "data/output/tomato-bed-tiny"):
    """Download results from Modal volume."""
    print(f"📦 Downloading from Modal volume to {local_dir}...")
    download_results.remote(remote_dir, local_dir)
