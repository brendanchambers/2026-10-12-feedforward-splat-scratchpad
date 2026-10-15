"""
Simple feedforward 3D Gaussian Splatting inference using WorldMirror.

This script runs on Modal cloud GPUs to generate .ply Gaussian Splat files
from input images. It's designed as the simplest possible example.
"""

import sys
from pathlib import Path

import modal

# Modal app configuration
app = modal.App("feedforward-3dgs")

# Create Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libgl1-mesa-glx",  # OpenGL library for opencv
        "libglib2.0-0",     # GLib library for opencv
        "libsm6",           # Session Management library
        "libxext6",         # X extensions library
        "libxrender1",      # X Rendering library
        "libgomp1",         # GNU OpenMP library
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2.0.0",
        "pillow",
        "tqdm",
        "huggingface-hub[torch]>=0.22",
        "einops",
        "roma",
        "trimesh",
        "plyfile",
        "opencv-python",
        "rich==13.3.5",
        "torchmetrics",
        "jaxtyping",
        "ninja",
        "onnxruntime",
        "pycolmap==3.10.0",
        "scipy",
        "matplotlib",
        "moviepy==1.0.3",
        "lpips",
        "viser",
        "colorspacious",
    )
    .pip_install(
        "gsplat",
        index_url="https://docs.gsplat.studio/whl/pt24cu124",
    )
    .run_commands(
        "git clone https://github.com/brendanchambers/2026-10-12-feedforward-splat-scratchpad.git /root/repo",
    )
)

# Mount local data directory
data_volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # 24GB GPU (T4 14GB insufficient for 16 images)
    timeout=3600,  # 1 hour timeout
    volumes={"/data": data_volume},
)
def run_inference(dataset_tag: str = "tomato-bed-tiny", verbose: bool = False, save_colmap: bool = True):
    """Run WorldMirror inference on input images with optional COLMAP output."""
    import subprocess
    import shutil

    print(f"Starting inference for dataset '{dataset_tag}'")

    # Setup paths
    input_path = Path("/data/incoming") / dataset_tag
    output_path = Path("/data/output") / dataset_tag

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy images to output/images directory (required by COLMAP format)
    images_output = output_path / "images"
    images_output.mkdir(exist_ok=True)

    print(f"Copying images from {input_path} to {images_output}")
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.copy2(img_file, images_output / img_file.name)

    # Change to repo directory
    import os
    os.chdir("/root/repo/repos/HunyuanWorld-Mirror")

    # Run official infer.py with COLMAP support
    cmd = [
        "python", "infer.py",
        "--input_path", str(images_output),
        "--output_path", str(output_path),
        "--save_gs",
        "--save_pointmap",
    ]

    if save_colmap:
        cmd.append("--save_colmap")

    if not verbose:
        cmd.append("--no-save_rendered")  # Skip video rendering for speed

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\n=== INFERENCE ERROR ===")
        print("STDOUT:", result.stdout)
        print("\nSTDERR:", result.stderr)
        print("=======================\n")
        raise RuntimeError(f"Inference failed with return code {result.returncode}")

    # Print output
    print(result.stdout)

    data_volume.commit()
    print("Inference complete!")


@app.local_entrypoint()
def main(dataset_tag: str = "tomato-bed-tiny", verbose: bool = False, save_colmap: bool = True):
    """Local entrypoint to run inference on Modal."""
    print(f"Launching Modal inference for dataset '{dataset_tag}'{' (verbose mode)' if verbose else ''}")
    run_inference.remote(dataset_tag=dataset_tag, verbose=verbose, save_colmap=save_colmap)
    print(f"Results will be in Modal volume at /data/output/{dataset_tag}")
