#!/bin/bash

# Timeout duration (e.g., 60 seconds)
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
timeout $timeout_duration python3 profiling_batch_flask.py &

# Wait until it is running
wait_for_flask

# Start the GPU utilization monitoring script in the background
echo "Starting GPU utilization monitoring"
timeout $timeout_duration bash monitor_gpu.sh &

# Start the inference script in the background
echo "Starting inference script"
timeout $timeout_duration python3 profiling_batch_calls.py &

# Wait for all background processes to complete
wait

echo "All processes have completed."
