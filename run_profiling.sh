#!/bin/bash

# Models
models="granite-7b,gemma-7b,llama3-8b"
# List of batch sizes to iterate
batch_sizes="1,2,3,4,5,7,8,11,13,16,19,23,29,32,36,41,48,53,56,60,64,68,72,79,83,86,89,94,100,105,109,112,120,128,136,149,167,191,211,233,256,307,353,409,457,512,1024,2048"
# Runs per batch size
num_runs=5


# Timeout duration
timeout_duration=10000

# Function to check if Flask API is up
wait_for_flask() {
  echo "Checking if Flask API is ready..."
  while ! curl -s http://127.0.0.1:5000/health >/dev/null; do
    echo "Waiting for Flask API to be ready"
    sleep 3
  done
  echo "Flask API is ready!"
}

# Start the Flask API in the background
echo "Starting Flask API"
timeout $timeout_duration python3 profiling_batch_flask.py $models &

# Wait until it is running
wait_for_flask

# Start the GPU utilization monitoring script in the background
echo "Starting GPU utilization monitoring"
timeout $timeout_duration bash monitor_gpu.sh &

# Start the inference script in the background
echo "Starting api calls script"
timeout $timeout_duration python3 profiling_batch_calls.py $models $batch_sizes $num_runs &

# Wait for all background processes to complete
wait

echo "All processes have been completed."
