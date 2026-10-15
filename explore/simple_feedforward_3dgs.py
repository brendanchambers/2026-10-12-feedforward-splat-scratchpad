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
        "rich",
        "torchmetrics",
        "jaxtyping",
        "ninja",
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
def run_inference(dataset_tag: str = "tomato-bed-tiny"):
    """Run WorldMirror inference on input images."""
    import glob
    import torch
    from PIL import Image

    # Add WorldMirror repo to path
    sys.path.insert(0, "/root/repo/repos/HunyuanWorld-Mirror")

    from src.models.models.worldmirror import WorldMirror
    from src.utils.inference_utils import prepare_images_to_tensor
    from src.utils.save_utils import save_gs_ply

    print(f"Starting inference for dataset '{dataset_tag}'")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model from HuggingFace
    print("Loading WorldMirror model...")
    model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    model.eval()

    # Load input images
    input_path = Path("/data/incoming") / dataset_tag
    img_paths = sorted(glob.glob(str(input_path / "*.jpg")) + glob.glob(str(input_path / "*.png")))

    if not img_paths:
        raise ValueError(f"No images found in {input_path}")

    print(f"Found {len(img_paths)} images")

    # Preprocess images
    imgs = prepare_images_to_tensor(img_paths, target_size=518, resize_strategy="crop").to(device)
    B, S, C, H, W = imgs.shape
    print(f"Image tensor shape: {imgs.shape}")

    # Prepare input
    views = {"img": imgs}
    cond_flags = [0, 0, 0]  # No conditioning [pose, depth, intrinsics]

    # TODO
    # add a debug flag
    # if --debug, visually verify the round trip image preprocessing.
    # for a random input image, show the original image, print its size in pixels
    # for the corresponding tensor held in imgs, visualize the tensor as a regular image, and print its size.
    # make it possible to compare the image before and after preprocessing, perhaps side by side.

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(views=views, cond_flags=cond_flags)

    # Save Gaussian Splat output
    output_path = Path("/data/output") / dataset_tag
    output_path.mkdir(parents=True, exist_ok=True)

    if "splats" in predictions:
        splats = predictions["splats"]
        means = splats["means"][0].reshape(-1, 3)
        scales = splats["scales"][0].reshape(-1, 3)
        quats = splats["quats"][0].reshape(-1, 4)
        opacities = splats["opacities"][0].reshape(-1)

        # Handle SH or colors
        if "sh" in splats:
            colors = splats["sh"][0].reshape(-1, 3)
        else:
            colors = splats["colors"][0].reshape(-1, 3)

        # Save PLY
        ply_path = output_path / "gaussians.ply"
        save_gs_ply(ply_path, means, scales, quats, colors, opacities)
        print(f"Saved Gaussian Splat to {ply_path}")

    # Also save point cloud if available
    if "pts3d" in predictions:
        from src.utils.save_utils import save_scene_ply

        pts_list = []
        colors_list = []

        for i in range(S):
            pts = predictions["pts3d"][0, i].reshape(-1, 3)
            img_colors = (imgs[0, i].permute(1, 2, 0) * 255).to(torch.uint8).reshape(-1, 3)
            pts_list.append(pts)
            colors_list.append(img_colors)

        all_pts = torch.cat(pts_list, dim=0)
        all_colors = torch.cat(colors_list, dim=0)

        ply_path = output_path / "pts_from_pointmap.ply"
        save_scene_ply(ply_path, all_pts, all_colors)
        print(f"Saved point cloud to {ply_path}")

    data_volume.commit()
    print("Inference complete!")


@app.local_entrypoint()
def main(dataset_tag: str = "tomato-bed-tiny"):
    """Local entrypoint to run inference on Modal."""
    print(f"Launching Modal inference for dataset '{dataset_tag}'")
    run_inference.remote(dataset_tag=dataset_tag)
    print(f"Results will be in Modal volume at /data/output/{dataset_tag}")
