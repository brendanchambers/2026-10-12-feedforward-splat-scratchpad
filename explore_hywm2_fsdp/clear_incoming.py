"""Clear incoming data directory for a dataset."""
import modal

app = modal.App("clear-incoming-data")
volume = modal.Volume.from_name("feedforward-3dgs-data")

@app.function(volumes={"/data": volume})
def clear_incoming(dataset_tag: str):
    import shutil
    from pathlib import Path

    path = Path("/data/incoming") / dataset_tag
    if path.exists():
        print(f"🗑️  Deleting {path}...")
        shutil.rmtree(path)
        volume.commit()
        print(f"✅ Deleted /data/incoming/{dataset_tag}")
    else:
        print(f"⏭️  Path does not exist: {path}")

@app.local_entrypoint()
def main(dataset_tag: str):
    clear_incoming.remote(dataset_tag)
