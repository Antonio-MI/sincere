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

run_duration = sys.argv[1] #120  # seconds
distribution = sys.argv[2] #"gamma" 

model_list = list(map(sys.argv[3].split(',')))#["granite-7b", "gemma-7b", "llama3-8b"] #["gpt2-124m", "distilgpt2-124m", "gpt2medium-355m"] 
model_frequencies = [0.1, 0.3, 0.6]

# Define the parameters for the Gamma distribution
# Use sys.argv
rate = 8
shape, scale = 1.0, 1/rate  # Shape (alpha) and scale (θ) for the Gamma distribution
# Rate 8 and shape 1 gives ~250 requests in 30 seconds | ~450 in 60 -> Close to an average of 8 request/s

# Define the API endpoint
api_url = "http://127.0.0.1:5000/inference"

# Folder where workload jsons are located
workload_folder = "./workloads"

count = 0

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

async def automated_calls(workloads, run_duration):
    count = 0
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < run_duration:
            # Calculate remaining time
            remaining_time = run_duration - (time.time() - start_time)
            if remaining_time <= 0:
                break

            # Select a random workload from the list
            workload = np.random.choice(workloads)
            
            # To change the frequency of each model, edit here:
            workload['model_alias'] = np.random.choice(model_list, p=model_frequencies) 
            print(workload['model_alias'])

            # Send the request asynchronously
            asyncio.create_task(send_request(session, workload))
            count += 1
            print(count)

            # Generate a sleep interval following the Gamma distribution
            if distribution == "gamma":
                interval = np.random.gamma(shape, scale)
            else:
                interval = 1
            print(f"Sleeping for {interval:.2f} seconds")

            # Adjust sleep time if it exceeds remaining time
            sleep_time = min(interval, remaining_time)
            await asyncio.sleep(sleep_time)

        # Send one last request with the fixed model_alias after the loop finishes
        final_workload = np.random.choice(workloads)
        final_workload['model_alias'] = "Stop"
        print(f"Sending final Stop order")
        await send_request(session, final_workload)

if __name__ == "__main__":
    try:
        workloads = load_workloads_from_folder(workload_folder)
        if not workloads:
            print("No valid JSON workloads found in the specified folder.")
        else:
            asyncio.run(automated_calls(workloads, run_duration))
    except KeyboardInterrupt:
        print("Process stopped")