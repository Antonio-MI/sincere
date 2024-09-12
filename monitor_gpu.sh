#!/bin/bash

# Generate a timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Output file for logging the GPU utilization with a timestamp in the filename
output_file="gpu_usage_${timestamp}.csv"

# Interval between each query (in seconds)
interval=1

# Run the query in a loop for a set period of time or until manually stopped
while true
do
  # Append the current timestamp and GPU utilization to the file
  echo "$(date), $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)" >> $output_file
  # Wait for the specified interval before querying again
  sleep $interval
done
