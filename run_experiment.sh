#!/bin/bash

# Timeout duration (e.g., 60 seconds)
timeout_duration=60
timeout_flask_duration=75

# Start the Flask API in the background
echo "Starting Flask API..."
timeout $timeout_flask_duration python3 profiling_batch_flask.py &

# Wait for Flask API to initialize (e.g., 5 seconds)
echo "Waiting for Flask API to initialize..."
sleep 15

# Start the GPU utilization monitoring script in the background
echo "Starting GPU utilization monitoring..."
timeout $timeout_duration bash monitor_gpu.sh &

# Start the inference script in the background
echo "Starting inference script..."
timeout $timeout_duration python3 profiling_batch_calls.py &

# Wait for all background processes to complete (optional)
wait

echo "All processes have completed."
