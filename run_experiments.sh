#!/bin/bash

# Run time for calls
run_duration=120
# Timeout duration (run of calls + 60 seconds)
timeout_duration=150
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
traffic_means=(4 8)
distributions=("gamma" "bursty" "ramp")
modes=("BestBatch" "BestBatch+Timer" "SelectBatch+Timer" "BestBatch+PartialBatch+Timer")
batch_time_limits=(40 60)


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
        echo "Starting GPU utilization monitoring"
        timeout $timeout_duration bash monitor_gpu.sh &

        # Start the inference script in the background
        echo "Starting API calls script with traffic_mean = $traffic_mean"
        timeout $timeout_duration python3 api_calls.py $run_duration $distribution $traffic_mean "$models" &

        # Wait for all background processes to complete
        wait

        echo "All processes for traffic_mean = $traffic_mean, batch_time_limit = $batch_time_limit, distribution = $distribution, mode = $mode have completed."
      done
    done
  done
done

echo "All experiments are done."


# #!/bin/bash

# # Run time for calls
# run_duration=1200
# # Timeout duration (run of calls + 60 seconds)
# timeout_duration=1260
# # Distribution followed by input calls
# distribution=bursty # gamma, bursty, ramp
# # Scheduling mode
# mode="SelectBatch+Timer" # One of [BestBatch, BestBatch+Timer, SelectBatch+Timer, BestBatch+PartialBatch+Timer]
# # Models
# models="granite-7b,gemma-7b,llama3-8b"
# # SLA
# batch_time_limit=40 # Set two SLAs 40 - 80s

# # Set quick runs for the whole grid of parameters

# # Function to check if Flask API is up
# wait_for_flask() {
#   echo "Checking if Flask API is ready..."
#   while ! curl -s http://127.0.0.1:5000/health >/dev/null; do
#     echo "Waiting for Flask API to be ready"
#     sleep 4
#   done
#   echo "Flask API is ready!"
# }

# # Iterate over traffic_mean values
# for traffic_mean in 4 8; do
#   echo "Running experiment with traffic_mean = $traffic_mean"

#   # Start the Flask API in the background
#   echo "Starting Flask API"
#   timeout $timeout_duration python3 api_scheduler_experiments.py $mode $models $batch_time_limit $distribution $run_duration $traffic_mean &

#   # Wait until it is running
#   wait_for_flask

#   # Start the GPU utilization monitoring script in the background
#   echo "Starting GPU utilization monitoring"
#   timeout $timeout_duration bash monitor_gpu.sh &

#   # Start the inference script in the background
#   echo "Starting api calls script with traffic_mean = $traffic_mean"
#   timeout $timeout_duration python3 api_calls.py $run_duration $distribution $traffic_mean $models &

#   # Wait for all background processes to complete
#   wait

#   echo "All processes for traffic_mean = $traffic_mean have completed."
# done

# echo "All experiments with different traffic_mean values are done."

##########################################

# #!/bin/bash



# # Run time for calls
# run_duration=900 
# # Timeout duration (run of calls + 60 seconds)
# timeout_duration=960
# # Distribution followed by input calls
# distribution=bursty #gamma, bursty, ramp
# # Mean requests per second for the patterns
# traffic_mean=2 #2, 5, 8
# # Scheduling mode
# mode="BestBatch+Timer" # One of [BestBatch, BestBatch+Timer, SelectBatch+Timer, BestBatch+PartialBatch, BestBatch+PartialBatch+Timer]
# # Models
# models="granite-7b,gemma-7b,llama3-8b"
# # SLA
# batch_time_limit=30

# # Function to check if Flask API is up
# wait_for_flask() {
#   echo "Checking if Flask API is ready..."
#   while ! curl -s http://127.0.0.1:5000/health >/dev/null; do
#     echo "Waiting for Flask API to be ready"
#     sleep 4
#   done
#   echo "Flask API is ready!"
# }

# # Start the Flask API in the background
# echo "Starting Flask API"
# timeout $timeout_duration python3 api_scheduler_experiments.py $mode $models $batch_time_limit $distribution $run_duration &

# # Wait until it is running
# wait_for_flask

# # Start the GPU utilization monitoring script in the background
# echo "Starting GPU utilization monitoring"
# timeout $timeout_duration bash monitor_gpu.sh &

# # Start the inference script in the background
# echo "Starting api calls script"
# timeout $timeout_duration python3 api_calls.py $run_duration $distribution $traffic_mean $models &

# # Wait for all background processes to complete
# wait

# echo "All processes have completed."
