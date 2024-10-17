#!/bin/bash

# Set the base directory and the output directory
BASE_DIR="/home/leh19/test_run_1/JPG/"
OUTPUT_DIR="/home/leh19/workspace/simple-image-comparison/results/"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Use rsync to copy the specific directories while preserving the hierarchy
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='transforms/' \
  --include='transforms/output/***' \
  --include='transforms_tight/' \
  --include='transforms_tight/output/***' \
  --include='transforms_tight/output_reduced/***' \
  --exclude='*' \
  "$BASE_DIR/" "$OUTPUT_DIR/"
