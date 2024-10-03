#!/bin/bash

# Run time for calls
run_duration=1200 #1200
# Timeout duration (run of calls + 60 seconds)
timeout_duration=1260 #1260
# Models
models="granite-7b,gemma-7b,llama3-8b"

# Function to check if Flask API is up
wait_for_flask() {
  echo "Checking if Flask API is ready..."
  while ! curl -s http://127.0.0.1:5000/health >/dev/null; do
    echo "Waiting for Flask API to be ready"
    sleep 4
  done
  echo "Flask API is ready!"
}

# Arrays of variables to iterate over
traffic_means=(12) # 12 ### not enough time for 16
distributions=("gamma" "bursty" "ramp")
modes=("BestBatch" "BestBatch+Timer" "SelectBatch+Timer" "BestBatch+PartialBatch+Timer")
batch_time_limits=(40 60 80) #40, 60, 80


# Iterate over batch_time_limits
for batch_time_limit in "${batch_time_limits[@]}"; do
  echo "Running experiments with batch_time_limit = $batch_time_limit"
  # Iterate over traffic_mean values
  for traffic_mean in "${traffic_means[@]}"; do
    echo "Running experiment with traffic_mean = $traffic_mean"
    # Iterate over distributions
    for distribution in "${distributions[@]}"; do
      echo "Running experiments for distribution: $distribution" 
      # Iterate over modes
      for mode in "${modes[@]}"; do
        echo "Running experiments for mode: $mode"

        # Start the Flask API in the background
        echo "Starting Flask API"
        timeout $timeout_duration python3 api_scheduler_experiments.py "$mode" "$models" $batch_time_limit $distribution $run_duration $traffic_mean &

        # Wait until it is running
        wait_for_flask

        # Start the GPU utilization monitoring script in the background
        # echo "Starting GPU utilization monitoring"
        # timeout $timeout_duration bash monitor_gpu.sh &

        # Start the inference script in the background
        echo "Starting API calls script with traffic_mean = $traffic_mean"
        timeout $timeout_duration python3 api_calls.py $run_duration $distribution $traffic_mean "$models" &

        # Wait for all background processes to complete
        wait

        echo "All processes for traffic_mean = $traffic_mean, batch_time_limit = $batch_time_limit, distribution = $distribution, mode = $mode have finished."
      done
    done
  done
done

echo "All experiments have finished."

