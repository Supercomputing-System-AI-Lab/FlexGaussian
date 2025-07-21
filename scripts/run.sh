#!/bin/bash

scene=$1

source_path="PATH/TO/DATASET"
model_path="PATH/TO/PRETRAINED/MODEL"
imp_score_path="PATH/TO/PRECALCULATED/IMPORTANCE/SCORE"

# Define all available scenes
all_scenes=(
    "bicycle"
    "bonsai"
    "counter"
    "kitchen"
    "room"
    "stump"
    "garden"
    "treehill"
    "flowers"
    "truck"
    "train"
    "drjohnson"
    "playroom"
)

# If user passed a scene as argument, use it; otherwise use all scenes
if [ -n "$scene" ]; then
    run_args=("$scene")
else
    run_args=("${all_scenes[@]}")
fi

# Loop over scenes and run
for s in "${run_args[@]}"; do
    echo "[INFO] Running on scene: $s"
    python compress.py \
        -s "$source_path/$s" \
        -m "$model_path/$s" \
        --output_path "output" \
        --quality_target_diff 1.0 \
        --eval \
        # --save_render \
        # --imp_score_path ${imp_score_path}
done

