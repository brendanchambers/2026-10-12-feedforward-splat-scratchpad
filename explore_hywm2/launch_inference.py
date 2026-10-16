"""
Feedforward 3D Gaussian Splatting inference using HunyuanWorld-Mirror 2.0.

This script runs on Modal cloud GPUs to generate .ply Gaussian Splat files
from input images using the HunyuanWorld-Mirror 2.0 model from the frozen
HY-World-2.0 repository.
"""

import sys
from pathlib import Path

import modal

# Modal app configuration
app = modal.App("feedforward-3dgs-hy-wm-2.0")


def download_models():
    """Download HY-World-2.0 model during image build to avoid runtime downloads."""
    import sys
    from pathlib import Path

    # Add frozen repo to Python path
    repo_path = Path("/root/repo/repos/HY-World-2.0")
    sys.path.insert(0, str(repo_path))

    print("[Build] Downloading HY-World-2.0 model from HuggingFace...")
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    # Download model to HuggingFace cache directory
    WorldMirrorPipeline.from_pretrained(
        "tencent/HY-World-2.0",
        subfolder="HY-WorldMirror-2.0",
        enable_bf16=True,
        disable_heads=["depth", "normal", "points"],
    )
    print("[Build] Model download complete!")


# Create Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
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
        "scipy",
        "matplotlib",
        "omegaconf",
        "safetensors",
        "packaging",
        "requests",
        "onnxruntime",
    )
    .pip_install(
        "flash-attn",
        index_url="https://wheels.astral.sh/simple/cu124/",
    )
    .pip_install(
        "gsplat",
        index_url="https://docs.gsplat.studio/whl/pt24cu124",
    )
    .run_commands(
        "git clone https://github.com/brendanchambers/2026-10-12-feedforward-splat-scratchpad.git /root/repo",
    )
    .run_function(download_models)
)

# Mount local data directory
data_volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": data_volume},
)
def run_inference(dataset_tag: str = "tomato-small-2", model_tag: str = "hy-wm-2.0"):
    """Run WorldMirror 2.0 inference on input images."""
    import os
    import shutil
    import sys

    # Reload volume to see latest uploaded data
    data_volume.reload()

    print(f"Starting HY-World-2.0 (WorldMirror 2.0) inference for dataset '{dataset_tag}'")

    # Setup paths
    input_path = Path("/data/incoming") / dataset_tag
    output_path = Path("/data/output") / model_tag / dataset_tag

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy images to output/images directory (pipeline expects them there)
    images_output = output_path
    images_output.mkdir(exist_ok=True)

    print(f"Copying images from {input_path} to {images_output}")
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            shutil.copy2(img_file, images_output / img_file.name)

    # Add frozen repo to Python path
    repo_path = Path("/root/repo/repos/HY-World-2.0")
    if not repo_path.exists():
        raise RuntimeError(f"Frozen repo not found at {repo_path}. Check clone command in image build.")

    sys.path.insert(0, str(repo_path))
    print(f"Using frozen HY-World-2.0 repo at {repo_path}")

    # Import pipeline from frozen repo
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline

    # Load the pipeline (will download model from HuggingFace)
    print("[Init] Loading WorldMirror 2.0 pipeline from tencent/HY-World-2.0...")
    pipeline = WorldMirrorPipeline.from_pretrained(
        "tencent/HY-World-2.0",
        subfolder="HY-WorldMirror-2.0",  # Note: Correct subfolder name
        enable_bf16=True,
        disable_heads=["depth", "normal", "points"],  # camera is required for gs
    )
    print("[Init] Pipeline loaded successfully")

    # Run inference with parameters matching the pipeline API
    print(f"[Inference] Processing images from {images_output}")
    result = pipeline(
        str(images_output),
        output_path=str(output_path),  # Use exact output path directly
        save_gs=True,
        save_points=False,
        save_camera=False,
        save_depth=False,
        save_normal=False,
        save_colmap=False,  # Skip COLMAP (not needed)
        log_time=True,
    )

    print("[Inference] Inference complete!")
    if result:
        print(f"[Save] Output directory: {result}")
    else:
        print("[Warning] Inference returned None (possibly skipped)")

    # Commit results
    data_volume.commit()
    print(f"[Save] Results saved to /data/output/{model_tag}/{dataset_tag}")


@app.local_entrypoint()
def main(dataset_tag: str = "tomato-small-2", model_tag: str = "hy-wm-2.0"):
    """Local entrypoint to run HY-World-2.0 inference on Modal."""
    print(f"Launching Modal inference for dataset '{dataset_tag}' with model '{model_tag}'")
    run_inference.remote(dataset_tag=dataset_tag, model_tag=model_tag)
    print(f"Results will be in Modal volume at /data/output/{model_tag}/{dataset_tag}")
