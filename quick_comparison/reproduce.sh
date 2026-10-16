
# Tomato-small-2 scene
# ~16 views of spring garden, camera views biased to one side.

# 
uv run --with modal modal run explore/pipeline.py \
    --dataset-tag tomato-small-2 \
    --model-tag hy-wm-1.1 \

# 
uv run --with modal modal run explore/pipeline.py \
    --dataset-tag tomato-small-2 \
    --model-tag hy-wm-2.0

# Tomato-small scene
# ~16 views of spring garden, camera views sample all sides


# 
uv run --with modal modal run explore/pipeline.py \
    --dataset-tag tomato-small \
    --model-tag hy-wm-1.1 \

# 
uv run --with modal modal run explore/pipeline.py \
    --dataset-tag tomato-small \
    --model-tag hy-wm-2.0



