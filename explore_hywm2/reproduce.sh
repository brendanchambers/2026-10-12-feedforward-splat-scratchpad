

# defaults
uv run --with modal modal run explore_hywm2/pipeline.py \
    --dataset-tag tomato-small \
    --model-tag hy-wm-2.0


# single gpu
uv run --with modal modal run explore_hywm2/pipeline.py \
    --dataset-tag tomato-small-2 \
    --model-tag hy-wm-2.0 \
    --gpu-mode 1gpu \
    --delete-data



# fsdp + seq parallel 
uv run --with modal modal run explore_hywm2/pipeline.py \
    --dataset-tag tomato-L \
    --model-tag hy-wm-2.0 \
    --gpu-mode fsdp \
    --frames_chunk_size 1 \
    --delete-data



