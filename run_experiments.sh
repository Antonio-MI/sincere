#!/bin/bash

# Run time for calls
run_duration=900 
# Timeout duration (run of calls + 60 seconds)
timeout_duration=960
# Distribution followed by input calls
distribution=bursty # gamma, bursty, ramp
# Scheduling mode
mode="BestBatch+PartialBatch" # One of [BestBatch, BestBatch+Timer, SelectBatch+Timer, BestBatch+PartialBatch, BestBatch+PartialBatch+Timer]
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

# Iterate over traffic_mean values
for traffic_mean in 2 5 8; do
  echo "Running experiment with traffic_mean = $traffic_mean"

  # Start the Flask API in the background
  echo "Starting Flask API"
  timeout $timeout_duration python3 api_scheduler_experiments.py $mode $models $batch_time_limit $distribution $run_duration $traffic_mean &

  # Wait until it is running
  wait_for_flask

  # Start the GPU utilization monitoring script in the background
  echo "Starting GPU utilization monitoring"
  timeout $timeout_duration bash monitor_gpu.sh &

  # Start the inference script in the background
  echo "Starting api calls script with traffic_mean = $traffic_mean"
  timeout $timeout_duration python3 api_calls.py $run_duration $distribution $traffic_mean $models &

  # Wait for all background processes to complete
  wait

  echo "All processes for traffic_mean = $traffic_mean have completed."
done

echo "All experiments with different traffic_mean values are done."

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
