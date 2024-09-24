import aiohttp
import asyncio
import time
import numpy as np
import json
import os
import random
import sys

# Seed for reproducible trials
np.random.seed(42)
random.seed(42)

# Try different frequency of use for each model

run_duration = int(sys.argv[1])     #120  # seconds
traffic_pattern = sys.argv[2]       #gamma, bursty, ramp
model_list = sys.argv[3].split(",")     #["granite-7b", "gemma-7b", "llama3-8b"] #["gpt2-124m", "distilgpt2-124m", "gpt2medium-355m"] 
print(model_list)
model_frequencies = [0.1, 0.3, 0.6]

# Define the API endpoint
api_url = "http://127.0.0.1:5000/inference"

# Folder where workload jsons are located
workload_folder = "./workloads"

def load_workloads_from_folder(folder):
    """Load all JSON files from the specified folder."""
    workload_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    workloads = []
    
    for file in workload_files:
        file_path = os.path.join(folder, file)
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)

                if 'model_alias' in data and 'prompt' in data:
                    filtered_payload = {
                        "model_alias": data["model_alias"],
                        "prompt": data["prompt"]
                    }
                    workloads.append(filtered_payload)
                else:
                    print(f"Missing required fields in {file_path}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {file_path}")
    return workloads

async def send_request(session, workload):
    try:
        async with session.post(api_url, json=workload, timeout=60) as response:
            if response.status == 200:
                result = await response.json()
                print(f"Response: {result}")
            else:
                print(f"Error: {response.status} - {await response.text()}")
    except Exception as e:
        #print(f"Failed to make the request: {e}")
        print()

async def automated_calls(workloads, run_duration, traffic_pattern):
    count = 0
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        # Initialize parameters based on traffic pattern
        if traffic_pattern == "gamma":
            # Parameters for Gamma distribution
            rate = 8
            shape, scale = 1.0, 1/rate  # Shape (alpha) and scale (θ)
            print("Starting Gamma traffic pattern")
        elif traffic_pattern == "bursty":
            # Parameters for Bursty Traffic
            # Initial burst and idle durations and rates
            burst_duration = np.random.uniform(1, 4)  # seconds
            idle_duration = np.random.uniform(8, 12)  # seconds
            burst_rate = np.random.uniform(30, 45)    # requests per second during burst
            idle_rate = np.random.uniform(0, 1)       # requests per second during idle
            # Initialize burst and idle periods
            is_burst = True
            period_start_time = start_time
            current_period_duration = burst_duration
            current_rate = burst_rate
            print(f"Starting with burst period for {current_period_duration:.2f} seconds at rate {current_rate:.2f} req/sec")
        elif traffic_pattern == "ramp":
            # Parameters for Repeating Ramp-up/Ramp-down Traffic
            min_rate = 1     # requests per second at start and end
            max_rate = 15    # peak requests per second
            ramp_up_duration = 20  
            ramp_down_duration = 20 
            cycle_duration = ramp_up_duration + ramp_down_duration
            print(f"Starting Repeating Ramp-up/Ramp-down traffic pattern with ramp_up_duration={ramp_up_duration:.2f}s and ramp_down_duration={ramp_down_duration:.2f}s")
            cycle_start_time = start_time
        else:
            print(f"Unknown traffic pattern '{traffic_pattern}'. Defaulting to constant rate.")
            traffic_pattern = "constant"
            interval = 0.25

        while time.time() - start_time < run_duration:
            # Calculate elapsed and remaining time
            elapsed_time = time.time() - start_time
            remaining_time = run_duration - elapsed_time
            if remaining_time <= 0:
                break

            # Select a random workload from the list
            workload = random.choice(workloads)
            
            # Adjust model frequency if needed
            workload['model_alias'] = np.random.choice(model_list, p=model_frequencies)
            print(f"Selected model_alias: {workload['model_alias']}")

            # Send the request asynchronously
            asyncio.create_task(send_request(session, workload))
            count += 1
            print(f"Request count: {count}")

            # Determine the sleep interval based on the traffic pattern
            if traffic_pattern == "gamma":
                interval = np.random.gamma(shape, scale)
            elif traffic_pattern == "bursty":
                # Check if we need to switch periods
                if time.time() - period_start_time >= current_period_duration:
                    # Switch between burst and idle periods
                    is_burst = not is_burst
                    period_start_time = time.time()
                    if is_burst:
                        # Re-sample burst parameters
                        burst_duration = np.random.uniform(1, 3)
                        burst_rate = np.random.uniform(15, 25)
                        current_period_duration = burst_duration
                        current_rate = burst_rate
                        print(f"Switching to burst period for {current_period_duration:.2f} seconds at rate {current_rate:.2f} req/sec")
                    else:
                        # Re-sample idle parameters
                        idle_duration = np.random.uniform(8, 15)
                        idle_rate = np.random.uniform(0, 1)
                        current_period_duration = idle_duration
                        current_rate = idle_rate
                        print(f"Switching to idle period for {current_period_duration:.2f} seconds at rate {current_rate:.2f} req/sec")

                # Generate inter-arrival time based on current rate
                interval = np.random.exponential(1 / current_rate) if current_rate > 0 else remaining_time
            elif traffic_pattern == "ramp":
                # Calculate current time within the current cycle
                time_in_cycle = (time.time() - cycle_start_time) % cycle_duration
                if time_in_cycle <= ramp_up_duration:
                    # Ramp up phase
                    current_rate = min_rate + (max_rate - min_rate) * (time_in_cycle / ramp_up_duration)
                else:
                    # Ramp down phase
                    time_in_ramp_down = time_in_cycle - ramp_up_duration
                    current_rate = max_rate - (max_rate - min_rate) * (time_in_ramp_down / ramp_down_duration)
                # Avoid division by zero
                current_rate = max(current_rate, 0.1)
                # Generate inter-arrival time based on current rate
                interval = np.random.exponential(1 / current_rate)
                print(f"Current rate: {current_rate:.2f} requests/sec")
            else:
                interval = 0.25  # Default sleep time

            print(f"Sleeping for {interval:.2f} seconds")

            # Adjust sleep time if it exceeds remaining time
            sleep_time = min(interval, remaining_time)
            await asyncio.sleep(sleep_time)

        # Send one last request to signal the end of the test
        final_workload = random.choice(workloads)
        final_workload['model_alias'] = "Stop"
        print(f"Sending final Stop request")
        await send_request(session, final_workload)

if __name__ == "__main__":
    try:
        workloads = load_workloads_from_folder(workload_folder)
        if not workloads:
            print("No valid JSON workloads found in the specified folder.")
        else:
            asyncio.run(automated_calls(workloads, run_duration, traffic_pattern))
    except KeyboardInterrupt:
        print("Process stopped")
