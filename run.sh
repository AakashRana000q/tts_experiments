#!/bin/bash

# Fixed value for N
N=256

# Array of valid functions
valid_functions=("dvts")

# Run for each function
for FUNCTION in "${valid_functions[@]}"; do
    echo "----------------------------------------"
    echo "Running function=$FUNCTION with n=$N"
    echo "----------------------------------------"
    
    if [ "$FUNCTION" == "beam_search" ]; then
        NUM_ITERATIONS=40
    else
        NUM_ITERATIONS=39
    fi

    python scripts/test_time_compute.py \
        "recipes/Llama-3.2-1B-Instruct/${FUNCTION}.yaml" \
        --n=$N \
        --num_samples=500 \
        --search_batch_size=4 \
        --num_iterations=$NUM_ITERATIONS \
        --push_to_hub=false \
        --lookahead=0 2>&1 | tee output_${FUNCTION}.log
        
    echo "Completed $FUNCTION"
    echo "----------------------------------------"
done

echo "All functions completed successfully!"