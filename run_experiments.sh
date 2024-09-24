#!/bin/bash



# Run time for calls
run_duration=120
# Timeout duration (run of calls + 30 seconds)
timeout_duration=150
# Distribution followed by input calls
distribution=gamma #gamma, bursty, ramp
# Scheduling mode
mode="BestBatch" # One of ["FCFS", "BatchedFCFS", "BestBatch", "BestBatch+Timer", "HigherBatch", "HigherBatch+Timer", "HigherBatch+PartialBatch+Timer"]
# Models
models="granite-7b,gemma-7b,llama3-8b"
# SLA
batch_time_limit=30

# Function to check if Flask API is up
wait_for_flask() {
  echo "Checking if Flask API is ready..."
  while ! curl -s http://127.0.0.1:5000/health >/dev/null; do
    echo "Waiting for Flask API to be ready"
    sleep 4
  done
  echo "Flask API is ready!"
}

# Start the Flask API in the background
echo "Starting Flask API"
timeout $timeout_duration python3 api_scheduler_experiments.py $mode $models $batch_time_limit &

# Wait until it is running
wait_for_flask

# Start the GPU utilization monitoring script in the background
echo "Starting GPU utilization monitoring"
timeout $timeout_duration bash monitor_gpu.sh &

# Start the inference script in the background
echo "Starting api calls script"
timeout $timeout_duration python3 api_calls.py $run_duration $distribution $models &

# Wait for all background processes to complete
wait

echo "All processes have completed."
