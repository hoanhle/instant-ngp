#!/bin/bash

# Define the paintings to process
PAINTINGS=(1 2 3 4 9)

# Define evaluation methods to loop through
EVAL_METHODS=("leave_one_out" "train_all")

# Path to your original script
SCRIPT_PATH="./experiment_scripts/train_paintings.sh"  # Replace with the actual path if needed

# Loop over each painting and evaluation method
for PAINTING in "${PAINTINGS[@]}"; do
    for EVAL_METHOD in "${EVAL_METHODS[@]}"; do
        echo "Processing painting $PAINTING with $EVAL_METHOD method using baseline..."
        bash "$SCRIPT_PATH" "$PAINTING" "$EVAL_METHOD" "baseline"
    done
done

echo "Batch processing completed!"
