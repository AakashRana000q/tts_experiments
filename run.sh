#!/bin/bash

# Fixed value for N
N=1024

# Array of valid functions
valid_functions=("beam_search" "best_of_n" "dvts" "dss")

# Run for each function
for FUNCTION in "${valid_functions[@]}"; do
    echo "----------------------------------------"
    echo "Running function=$FUNCTION with n=$N"
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