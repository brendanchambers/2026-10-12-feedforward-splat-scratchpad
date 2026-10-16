# HY-WorldMirror 2.0 Analysis

## Overview
HY-WorldMirror 2.0 is a feedforward 3D reconstruction model that takes multi-view images as input and outputs 3D Gaussian Splatting parameters, depth maps, normal maps, camera poses, and point clouds in a single forward pass.

## Architecture Components

### Core Model (`worldmirror.py`)
Located at: `repos/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/models/worldmirror.py`

**Key Architecture:**
- **Backbone**: VisualGeometryTransformer (vision transformer-based)
- **Model Sizes**: small, base, large, giant
  - Large (default): 1024 embed_dim, 24 layers, 16 heads
  - Supports distributed inference via sequence parallelism
- **Multi-head outputs**:
  - Camera head (poses & intrinsics)
  - Depth head (DPT-based dense prediction)
  - Normal head
  - Points head
  - Gaussian Splatting head (256-dim features)

**Key Features:**
- Conditional inputs: can accept prior camera poses, depth, or intrinsics
- RoPE (Rotary Position Embeddings) with normalization
- bf16 precision support with selective fp32 for critical layers
- FSDP (Fully Sharded Data Parallel) for multi-GPU
- Patch-based processing (14x14 patches by default)

---

## Pipeline (`pipeline.py`)
Located at: `repos/HY-World-2.0/hyworld2/worldrecon/pipeline.py`

**Purpose**: Production inference wrapper with comprehensive I/O handling

### Key Capabilities:

1. **Model Loading**
   - HuggingFace Hub integration (`tencent/HY-World-2.0`)
   - Local checkpoint support (.safetensors, .ckpt)
   - Automatic model file resolution

2. **Inference Modes**
   - Single GPU: straightforward forward pass
   - Multi-GPU: FSDP-based sequence parallel across views
   - Supports bf16 mixed precision
   - CPU offload for memory efficiency

3. **Input Processing**
   - Images: PNG, JPG, JPEG, WEBP
   - Video: automatic frame extraction at specified FPS
   - Adaptive resolution (default max 952px)
   - Optional prior conditioning (camera, depth)

4. **Output Generation**
   ```
   output_dir/
   ├── depth/           # depth maps per view
   ├── normal/          # normal maps per view
   ├── camera_params.json
   ├── points.ply       # point cloud
   ├── gaussians.ply    # 3D gaussian splats
   ├── colmap/          # (optional) COLMAP format
   └── rendered/        # (optional) interpolated video
   ```

5. **Post-processing**
   - Sky masking (model-based or ONNX)
   - Edge filtering (depth/normal discontinuities)
   - Confidence filtering
   - Voxel-based gaussian pruning
   - Point cloud downsampling

6. **CLI Interface**
   ```bash
   # Single GPU
   python -m hyworld2.worldrecon.pipeline --input_path path/to/images

   # Multi-GPU
   torchrun --nproc_per_node=2 -m hyworld2.worldrecon.pipeline \
       --input_path path/to/images --use_fsdp --enable_bf16
   ```

---

## Gradio App (`gradio_app.py`)
Located at: `repos/HY-World-2.0/hyworld2/worldrecon/gradio_app.py`

**Purpose**: Interactive web-based demo interface

### Features:

1. **User Interface**
   - File upload (images/video)
   - Video frame extraction with adjustable sampling interval
   - Real-time terminal output capture
   - Multi-tab visualization:
     - 3D Gaussian Splatting viewer
     - Point cloud/mesh viewer
     - Depth map browser (per-view navigation)
     - Normal map browser
     - Camera parameter download

2. **Interactive Controls**
   - Frame selector (show specific frame's points)
   - Reconstruction options:
     - Show/hide camera frustums
     - Mesh vs point cloud rendering
     - Filter low-confidence/edge regions
     - Sky background filtering
   - Example scene gallery (realistic/stylistic)

3. **Multi-GPU Support**
   - Rank 0: hosts Gradio server
   - Workers: wait for broadcast inference commands
   - String-based IPC for task distribution

4. **Processing Flow**
   ```
   Upload → Extract/Resample → Reconstruct → Visualize
                                    ↓
                            (GLB scene + PLY gaussians)
   ```

5. **Visualization Outputs**
   - GLB files for 3D viewers (with configurable options)
   - PLY gaussians (pruned & downsampled to 5M max)
   - Depth/normal as colored images (turbo colormap)

6. **Launch**
   ```bash
   python -m hyworld2.worldrecon.gradio_app \
       --examples_dir ./examples/worldrecon \
       --port 8081
   ```

---

## Comparison: Pipeline vs Gradio App

| Aspect | Pipeline | Gradio App |
|--------|----------|------------|
| **Use Case** | Batch processing, production | Interactive exploration, demo |
| **Interface** | CLI / Python API | Web browser |
| **Input** | File paths | Upload widget |
| **Output** | Comprehensive (all formats) | Focused (GLB + PLY) |
| **Interactivity** | None (run & done) | Live parameter changes |
| **Visualization** | Files only | Embedded 3D viewers |
| **Multi-GPU** | Full FSDP support | Basic sequence parallel |

---

## Key Inference Flow

1. **Image Loading**: `prepare_images_to_tensor()` → [B, S, 3, H, W]
2. **Preprocessing**: resize/crop to target_size (adaptive)
3. **Optional Conditioning**: load prior camera/depth
4. **Forward Pass**: WorldMirror model → predictions dict
   - depth: [B, S, 1, H, W]
   - normals: [B, S, 3, H, W]
   - camera_poses: [B, S, 4, 4]
   - camera_intrs: [B, S, 3, 3]
   - splats: {means, scales, quats, opacities, sh/colors}
5. **Post-processing**: masking, filtering, compression
6. **Saving**: multiple output formats

---

## Technical Highlights

### Memory Efficiency
- Selective bf16 casting (preserve fp32 for critical layers)
- FSDP with optional CPU offload
- Voxel-based gaussian pruning
- Configurable output compression

### Robustness
- Adaptive resolution based on input aspect ratios
- Sky detection (multiple methods)
- Edge/confidence filtering
- Handles 1-32 frames per inference

### Flexibility
- Disable unused heads to save memory
- Prior conditioning support
- COLMAP export for downstream tools
- Video rendering from gaussians

---

## Model Files Expected

```
tencent/HY-World-2.0/
└── HY-WorldMirror-2.0/
    ├── model.safetensors  # weights
    └── config.yaml        # model config (or config.json)
```

Resolution order:
1. Local path with subfolder
2. Local path direct
3. HuggingFace Hub download

---

## Summary

**Pipeline** is the production workhorse:
- Comprehensive output formats
- Robust CLI with all parameters exposed
- Optimized for batch processing and automation
- Full multi-GPU support with FSDP

**Gradio App** is the interactive playground:
- User-friendly web interface
- Real-time visualization
- Quick iteration on reconstruction parameters
- Great for demos and exploration

Both share the same core `WorldMirrorPipeline._run_inference()` method, ensuring consistency between interactive and batch modes.
