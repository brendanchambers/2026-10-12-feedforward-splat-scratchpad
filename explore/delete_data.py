"""
Delete dataset data from Modal volume.

Use this to clean up incoming and output data for a fresh run.
"""

import modal

app = modal.App("feedforward-3dgs-delete")
volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(volumes={"/data": volume})
def delete_dataset_data(dataset_tag: str):
    """Delete incoming and output data for a dataset from the volume."""
    import shutil
    from pathlib import Path

    paths_to_delete = [
        Path("/data/incoming") / dataset_tag,
        Path("/data/output") / dataset_tag,
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
def main(dataset_tag: str):
    """Delete data for a dataset from Modal volume."""
    print(f"🗑️  Deleting data for dataset '{dataset_tag}' from Modal volume...")
    deleted = delete_dataset_data.remote(dataset_tag)
    if deleted:
        print(f"\n✅ Successfully deleted:")
        for path in deleted:
            print(f"   - {path}")
    else:
        print(f"\nℹ️  No data found for dataset '{dataset_tag}'")
