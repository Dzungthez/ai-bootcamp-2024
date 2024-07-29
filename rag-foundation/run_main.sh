#!/bin/bash

DATA_PATH="qasper-test-v0.3.json"
OUTPUT_PATH="predictions.jsonl"

# Set other parameters
MODE="sparse"
FORCE_INDEX=false
PRINT_CONTEXT=false
CHUNK_SIZE=200
TOP_K=5
RETRIEVAL_ONLY=false

COMMAND="python scripts/main.py --data_path $DATA_PATH --output_path $OUTPUT_PATH --mode $MODE --force_index $FORCE_INDEX --print_context $PRINT_CONTEXT --chunk_size $CHUNK_SIZE --top_k $TOP_K --retrieval_only $RETRIEVAL_ONLY"


echo "Running command: $COMMAND"

eval $COMMAND
