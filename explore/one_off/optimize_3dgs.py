"""
Post-3DGS optimization using gsplat simple_trainer_worldmirror.

This script runs posthoc optimization on the initial 3DGS output to improve quality.
Results are saved to a separate 'optimized' subdirectory.
"""

import sys
from pathlib import Path

import modal

app = modal.App("feedforward-3dgs-optimize")

# Create Modal image with gsplat training dependencies
# Use micromamba image for CUDA support (provides nvcc for extension compilation)
image = (
    modal.Image.micromamba(python_version="3.10")
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "libgomp1",
        "build-essential",
    )
    .micromamba_install(
        "cudatoolkit=11.8",  # CUDA 11.8 for PyTorch cu118
        "cuda-nvcc=11.8",    # CUDA compiler matching PyTorch
        channels=["nvidia", "conda-forge"],
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu118",  # Match CUDA 11.8
    )
    .pip_install(
        "numpy<2.0.0",
        "pillow",
        "tqdm",
        "opencv-python",
        "einops",
        "roma",
        "trimesh",
        "plyfile",
        "rich==13.3.5",
        "torchmetrics[image]",
        "jaxtyping",
        "ninja",
        "onnxruntime",
        "scipy",
        "matplotlib",
        "moviepy==1.0.3",
        "lpips",
        "viser",
        "colorspacious",
        # Additional deps for simple_trainer_worldmirror
        "tyro>=0.8.8",
        "tensorboard",
        "pyyaml",
        "imageio[ffmpeg]",
        "scikit-learn",
        "tensorly",
        "splines",
    )
    .pip_install(
        "gsplat",
        index_url="https://docs.gsplat.studio/whl/pt24cu118",  # Match CUDA 11.8
    )
    .run_commands(
        "git clone https://github.com/brendanchambers/2026-10-12-feedforward-splat-scratchpad.git /root/repo",
    )
)

data_volume = modal.Volume.from_name("feedforward-3dgs-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hour timeout for optimization
    volumes={"/data": data_volume},
)
def run_optimization(dataset_tag: str = "tomato-bed-tiny", max_steps: int = 3000):
    """Run post-3DGS optimization on inference outputs."""
    import subprocess
    import os

    # Reload volume to see latest inference data
    data_volume.reload()

    print(f"Starting optimization for dataset '{dataset_tag}'")

    # Install CUDA extension packages at runtime (when GPU is available)
    # Set CUDA_HOME environment variable for CUDA extension compilation
    # Micromamba installs CUDA toolkit to /opt/conda
    cuda_path = "/opt/conda"
    os.environ["CUDA_HOME"] = cuda_path
    os.environ["CUDA_PATH"] = cuda_path
    # Add CUDA library paths
    os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib:{cuda_path}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    # Add CUDA bin to PATH
    os.environ["PATH"] = f"{cuda_path}/bin:" + os.environ.get("PATH", "")
    print(f"Set CUDA_HOME={cuda_path}")

    # Verify nvcc is available
    nvcc_test = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
    if nvcc_test.returncode == 0:
        print(f"nvcc found at: {nvcc_test.stdout.strip()}")
        # Verify CUDA version
        nvcc_version = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(f"nvcc version:\n{nvcc_version.stdout}")
    else:
        print("WARNING: nvcc not found in PATH")

    print("Installing CUDA extensions (nerfview, fused-ssim, fused-bilagrid)...")
    print(f"Environment: CUDA_HOME={os.environ['CUDA_HOME']}, PATH includes {cuda_path}/bin")

    # Install packages with optional ones that can fail
    packages = [
        ("nerfview", "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787", True),  # Required
        ("pycolmap", "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e", True),  # Required - provides pycolmap2
        ("fused-ssim", "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5", False),  # Optional - performance optimization
        ("fused-bilagrid", "git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe", False),  # Optional - tone mapping
    ]

    for pkg_name, pkg_url, required in packages:
        print(f"\nInstalling {pkg_name}{'(required)' if required else '(optional)'}...")
        install_cmd = ["pip", "install", pkg_url]
        env = os.environ.copy()
        result = subprocess.run(install_cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            if required:
                print(f"\n❌ {pkg_name} installation FAILED (required)")
                print(f"\n=== ERROR OUTPUT ===")
                print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
                raise RuntimeError(f"Failed to install required package {pkg_name}: {result.returncode}")
            else:
                print(f"⚠️  {pkg_name} installation FAILED (optional) - continuing without it")
                print(f"   This may reduce performance but won't prevent optimization from running")
        else:
            print(f"✅ {pkg_name} installed successfully")

    print("\n✅ Required CUDA extensions installed successfully")

    # Patch 1: colmap.py to use pycolmap instead of pycolmap2
    # The rmbrualla fork should provide pycolmap2, but it actually installs as pycolmap
    colmap_py_path = "/root/repo/repos/HunyuanWorld-Mirror/submodules/gsplat/examples/datasets/colmap.py"
    print(f"Patching {colmap_py_path} to use 'pycolmap' instead of 'pycolmap2'...")
    with open(colmap_py_path, "r") as f:
        content = f.read()
    content = content.replace("from pycolmap2 import", "from pycolmap import")
    with open(colmap_py_path, "w") as f:
        f.write(content)
    print("✅ Patched colmap.py")

    # Patch 2: simple_trainer_worldmirror.py to make fused_ssim optional
    trainer_py_path = "/root/repo/repos/HunyuanWorld-Mirror/submodules/gsplat/examples/simple_trainer_worldmirror.py"
    print(f"Patching {trainer_py_path} to make fused_ssim optional...")
    with open(trainer_py_path, "r") as f:
        content = f.read()

    # Replace the import with a try/except
    content = content.replace(
        "from fused_ssim import fused_ssim",
        "try:\n    from fused_ssim import fused_ssim\n    HAS_FUSED_SSIM = True\nexcept ImportError:\n    from torchmetrics.functional import structural_similarity_index_measure as ssim\n    HAS_FUSED_SSIM = False\n    print('Warning: fused_ssim not available, using torchmetrics SSIM (slower)')"
    )

    # Replace the multi-line fused_ssim call with conditional
    old_ssim_code = """l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )"""

    new_ssim_code = """l1loss = F.l1_loss(colors, pixels)
            if HAS_FUSED_SSIM:
                ssimloss = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
                )
            else:
                ssimloss = 1.0 - ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                )"""

    content = content.replace(old_ssim_code, new_ssim_code)

    with open(trainer_py_path, "w") as f:
        f.write(content)
    print("✅ Patched simple_trainer_worldmirror.py")

    # Setup paths
    output_dir = Path("/data/output") / dataset_tag
    data_dir = output_dir / "images"  # gsplat expects COLMAP data in images/sparse/
    result_dir = output_dir / "optimized"

    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")

    if not data_dir.exists():
        raise ValueError(f"Images directory does not exist: {data_dir}")

    # Check for required COLMAP data
    sparse_dir = data_dir / "sparse"
    if not sparse_dir.exists():
        raise ValueError(
            f"COLMAP sparse directory not found at {sparse_dir}. "
            "Make sure inference was run with COLMAP output enabled."
        )

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Add gsplat examples to path
    sys.path.insert(0, "/root/repo/repos/HunyuanWorld-Mirror/submodules/gsplat/examples")

    # Change to gsplat examples directory (required for relative imports)
    import os
    os.chdir("/root/repo/repos/HunyuanWorld-Mirror/submodules/gsplat/examples")

    # Run optimization
    print(f"Running optimization for {max_steps} steps...")
    cmd = [
        "python",
        "simple_trainer_worldmirror.py",
        "default",
        "--data_factor", "1",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(max_steps),
        "--disable_viewer",
        "--save_ply",
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Optimization failed with return code {result.returncode}")

    print(result.stdout)

    # Commit changes to volume
    data_volume.commit()
    print(f"Optimization complete! Results saved to {result_dir}")


@app.local_entrypoint()
def main(dataset_tag: str = "tomato-bed-tiny", max_steps: int = 3000):
    """Local entrypoint to run optimization on Modal."""
    print(f"Launching Modal optimization for dataset '{dataset_tag}' ({max_steps} steps)")
    run_optimization.remote(dataset_tag=dataset_tag, max_steps=max_steps)
    print(f"Results will be in Modal volume at /data/output/{dataset_tag}/optimized")
