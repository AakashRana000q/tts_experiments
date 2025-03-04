#!/bin/bash

# Default values
MODE="full"
FUNCTION="dss"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode=*)
        MODE="${1#*=}"
        shift
        ;;
        --function=*)
        FUNCTION="${1#*=}"
        shift
        ;;
        *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done

# Validate function name
valid_functions=("beam_search" "best_of_n" "dvts" "dss")
if [[ ! " ${valid_functions[@]} " =~ " ${FUNCTION} " ]]; then
    echo "Error: Invalid function. Must be one of: ${valid_functions[*]}"
    exit 1
fi

# Set n based on mode
if [ "$MODE" == "test" ]; then
    N=50
elif [ "$MODE" == "full" ]; then
    N=1024
else
    echo "Error: Mode must be either 'test' or 'full'"
    exit 1
fi

# Run the Python script
echo "Running with mode=$MODE, function=$FUNCTION, n=$N"
# python scripts/test_time_compute.py \
#     "recipes/Llama-3.2-1B-Instruct/${FUNCTION}.yaml" \
#     --n=$N \
#     --num_samples=500 \
#     --push_to_hub=true \
#     --hub_dataset_private=true \
#     --lookahead=0