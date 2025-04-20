#!/bin/bash

# Array of valid functions
valid_functions=("data_pb")

# Run for each function
for FUNCTION in "${valid_functions[@]}"; do
    echo "----------------------------------------"
    echo "Running function=$FUNCTION "
    echo "----------------------------------------"
    
    python scripts/create_data.py \
        "recipes/Llama-3.2-1B-Instruct/${FUNCTION}.yaml" \
        --beam_width=8 \
        --search_batch_size=256 \
        --push_to_hub=false \
        --lookahead=0 2>&1 | tee output_${FUNCTION}.log
        
    echo "Completed $FUNCTION"
    echo "----------------------------------------"
done

echo "All functions completed successfully!"