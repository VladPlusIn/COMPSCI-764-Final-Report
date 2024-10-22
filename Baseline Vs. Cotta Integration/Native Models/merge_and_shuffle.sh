#!/bin/bash

# Define base folder paths
BASE_PATH=""
OUTPUT_FOLDER="$BASE_PATH/total"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# List of dataset folder names
datasets=("1458" "2259" "2261" "2821" "2997" "3358" "3386" "3427" "3476")

# For totaltrain.log.txt
first=1
for dataset in "${datasets[@]}"; do
    if [ $first -eq 1 ]; then
        # Add header from the first file
        cat "$BASE_PATH/$dataset/train.log.txt" > "$OUTPUT_FOLDER/totaltrain.log.txt"
        first=0
    else
        # Skip the first line (header) from subsequent files
        tail -n +2 "$BASE_PATH/$dataset/train.log.txt" >> "$OUTPUT_FOLDER/totaltrain.log.txt"
    fi
done

# For totaltest.log.txt
first=1
for dataset in "${datasets[@]}"; do
    if [ $first -eq 1 ]; then
        # Add header from the first file
        cat "$BASE_PATH/$dataset/test.log.txt" > "$OUTPUT_FOLDER/totaltest.log.txt"
        first=0
    else
        # Skip the first line (header) from subsequent files
        tail -n +2 "$BASE_PATH/$dataset/test.log.txt" >> "$OUTPUT_FOLDER/totaltest.log.txt"
    fi
done