#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 <painting_number> <eval_method> <method_option>"
    echo "  painting_number: e.g., 2 (for painting_2)"
    echo "  eval_method: leave_one_out or train_all"
    echo "  method_option: baseline or our"
    exit 1
}

# Ensure the correct number of arguments
if [ $# -ne 3 ]; then
    usage
fi

# Assign arguments
PAINTING_NUM=$1
EVAL_METHOD=$2  # "leave_one_out" or "train_all"
METHOD_OPTION=$3  # "baseline" or "our"

# Base dataset directory
BASE_DIR="/home/leh19/datasets/paintings/test_run_1/JPG"

# Select config file based on method_option
if [ "$METHOD_OPTION" == "our" ]; then
    CONFIG_FILE="configs/nerf/reduced.json"
else
    CONFIG_FILE="configs/nerf/base.json"
fi

# Construct paths dynamically
SCENE_PATH="$BASE_DIR/painting_${PAINTING_NUM}/${EVAL_METHOD}/${METHOD_OPTION}/train.json"
TEST_TRANSFORMS_PATH="$BASE_DIR/painting_${PAINTING_NUM}/${EVAL_METHOD}/${METHOD_OPTION}/test.json"
OUTPUT_DIR="$BASE_DIR/painting_${PAINTING_NUM}/${EVAL_METHOD}/${METHOD_OPTION}/"
SNAPSHOT_PATH="$OUTPUT_DIR/${METHOD_OPTION}.ingp"

# Ensure the required files exist
if [ ! -f "$SCENE_PATH" ] || [ ! -f "$TEST_TRANSFORMS_PATH" ]; then
    echo "Error: One or more required files are missing."
    echo "Check paths:"
    echo "  SCENE_PATH: $SCENE_PATH"
    echo "  TEST_TRANSFORMS_PATH: $TEST_TRANSFORMS_PATH"
    exit 1
fi


# Run the command
echo "Running NeRF training with the following parameters:"
echo "  Painting Number: $PAINTING_NUM"
echo "  Evaluation Method: $EVAL_METHOD"
echo "  Method Option: $METHOD_OPTION"
echo "  Config File: $CONFIG_FILE"
echo "  Scene Path: $SCENE_PATH"
echo "  Test Transforms: $TEST_TRANSFORMS_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Save Snapshot: $SNAPSHOT_PATH"

python3 scripts/run.py "$CONFIG_FILE" \
    --scene "$SCENE_PATH" \
    --test_transforms "$TEST_TRANSFORMS_PATH" \
    --n_steps 35000 \
    --output_dir "$OUTPUT_DIR" \
    --save_snapshot "$SNAPSHOT_PATH"
