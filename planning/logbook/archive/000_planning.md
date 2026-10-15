lan: Feedforward 3DGS Inference with WorldMirror

  Based on my exploration, here's the proposed plan:

  Current State

  - WorldMirror repo: Already cloned at repos/HunyuanWorld-Mirror
  - Input data: 16 JPG frames in data/incoming/tomato-bed-tiny/
  - Target: Generate .ply Gaussian Splat file from these images
  - Environment: M4 Apple Air (local) + Modal for GPU (if needed)

  Proposed Implementation Plan

  1. Environment Setup
  - Add WorldMirror dependencies to pyproject.toml (PyTorch, gsplat, etc.)
  - Install dependencies using uv
  - Verify CUDA availability (or fallback to CPU/MPS for Mac)

  2. Create Inference Script (main.py)
  - Import WorldMirror model from the repo
  - Load pretrained weights from HuggingFace (tencent/HunyuanWorld-Mirror)
  - Prepare input images from data/incoming/tomato-bed-tiny
  - Run inference to generate:
    - Point clouds
    - Depth maps
    - Camera parameters
    - Gaussian splats (.ply file)
  - Save outputs to data/output/tomato-bed-tiny/

  3. Output Structure
  data/output/tomato-bed-tiny/
  ├── gaussians.ply          # Main output - 3D Gaussian Splat
  ├── pts_from_pointmap.ply  # Point cloud
  ├── rendered.mp4           # Rendered views
  ├── depth/                 # Depth maps
  └── normal/                # Normal maps

  4. Visualization
  - Use a web-based Gaussian Splat viewer (e.g., gsplat viewer or antimatter15's viewer)
  - Either launch locally or provide instructions to view the .ply file

  Key Questions

  1. Compute environment: Should we run locally on M4 (CPU/MPS) or set up Modal for GPU? The M4 might
   be slow for inference.
  2. Model weights: The script will download from HuggingFace automatically - is internet access
  available?
  3. Image preprocessing: The 16 frames appear to be sequential. Should we:
    - Use all 16 frames?
    - Subsample them?
    - Any specific FPS or ordering requirements?
  4. Viewer preference: Which web viewer would you prefer?
    - SuperSplat (browser-based)
    - antimatter15's viewer
    - gsplat's built-in viewer
    - Other?
  5. Design approach: Following the style guide emphasis on brevity - should I create a single
  concise main.py that handles everything, or separate concerns into multiple files?

  Would you like me to proceed with this plan? Any preferences on the questions above?

User response begins below:
Let's refine the plan before we move forward. Here is my feedback:
(1) We will not use CUDA or MPS locally, we will use Modal for an on-demand cloud gpu. We will use as inexpensive a gpu as is feasible since usage is metered. 
(2) Begin by using all 16 frames. We can adjust if needed. No order requirement. 
(3) Let's wait to implement any visualization. For now I will use a web viewer. 
(4) Use a single script, meant as a simplest example. The script should be located at
`explore/simple_feedforward_3dgs.py`. Please incorporate my feedback into the planning document you
created above. Then write our updated plan to `planning/logbook/001_planning.md`