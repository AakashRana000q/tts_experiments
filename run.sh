#!/bin/bash

# Fixed value for N
N=256

# Array of valid functions
valid_functions=("dis")

# Run for each function
for FUNCTION in "${valid_functions[@]}"; do
    echo "----------------------------------------"
    echo "Running function=$FUNCTION with n=$N"
    echo "----------------------------------------"
    
    python scripts/test_time_compute.py \
        "recipes/Llama-3.2-1B-Instruct/${FUNCTION}.yaml" \
        --n=$N \
        --num_samples=500 \
        --push_to_hub=false \
        --lookahead=0 2>&1 | tee output_${FUNCTION}.log
        
    echo "Completed $FUNCTION"
    echo "----------------------------------------"
done

echo "All functions completed successfully!"