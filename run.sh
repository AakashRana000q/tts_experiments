#!/bin/bash

# Default values
MODE="full"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode=*)
        MODE="${1#*=}"
        shift
        ;;
        *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done

# Set n based on mode
if [ "$MODE" == "test" ]; then
    N=50
elif [ "$MODE" == "full" ]; then
    N=1024
else
    echo "Error: Mode must be either 'test' or 'full'"
    exit 1
fi

# Array of valid functions
valid_functions=("beam_search" "dvts" "dss")

# Run for each function
for FUNCTION in "${valid_functions[@]}"; do
    echo "----------------------------------------"
    echo "Running with mode=$MODE, function=$FUNCTION, n=$N"
    echo "----------------------------------------"
    
    python scripts/test_time_compute.py \
        "recipes/Llama-3.2-1B-Instruct/${FUNCTION}.yaml" \
        --n=$N \
        --num_samples=500 \
        --push_to_hub=true \
        --hub_dataset_private=true \
        --lookahead=0
        
    echo "Completed $FUNCTION"
    echo "----------------------------------------"
done

echo "All functions completed successfully!"