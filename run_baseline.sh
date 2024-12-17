#!/bin/bash

# Define the list of models to run
models=("resnet50" "resnet101" "densenet121" "mobilenet_v2" "efficientnet_b0" "inception_v3" "mobilenet_v2_from_scratch")
# models=("resnet50" "resnet101" )
# models=("densenet121" "mobilenet_v2" "efficientnet_b0" "inception_v3" "mobilenet_v2_from_scratch" )

# Loop through the list and run baseline.py with each model
for model in "${models[@]}"
do
    echo "Running baseline.py with model: $model"
    python baseline.py --baseline_model "$model"
done
