import aiohttp
import asyncio
import time
import numpy as np
import json
import os
import random

# Seed for reproducible trials
np.random.seed(42)
random.seed(42)

# Define the API endpoint
api_url = "http://127.0.0.1:5000/inference"

# Folder where workload jsons are located
workload_folder = "./workloads"

# List of batch sizes to profile
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# List of models to profile
models_to_profile = ["gpt2-124m", "distilgpt2-124m", "gptneo-125m"]  # Add more models as needed

# Number of profiling runs per batch size
num_runs_per_batch_size = 5

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
        async with session.post(api_url, json=workload) as response:
            if response.status == 200:
                result = await response.json()
                print(f"Response: {result}")
            else:
                print(f"Error: {response.status} - {await response.text()}")
    except Exception as e:
        print(f"Failed to make the request: {e}")

async def profile_batch_size(session, workloads, model, batch_size):
    tasks = []
    for _ in range(batch_size):
        # Select a random workload from the list
        workload = np.random.choice(workloads)
        workload['model_alias'] = model  # Set the current model alias
        workload['batch_size'] = batch_size  # Add the batch size to the workload

        # Send the request asynchronously
        tasks.append(asyncio.create_task(send_request(session, workload)))

    # Wait for all tasks in this batch size to complete
    await asyncio.gather(*tasks)

async def automated_batch_profiling(workloads):
    async with aiohttp.ClientSession() as session:
        for model in models_to_profile:
            print(f"Profiling model: {model}")
            for batch_size in batch_sizes:
                print(f"  Profiling batch size: {batch_size}")
                for run in range(num_runs_per_batch_size):
                    print(f"    Run {run + 1} for batch size {batch_size}")
                    await profile_batch_size(session, workloads, model, batch_size)
                    await asyncio.sleep(2)  # Small delay between runs

if __name__ == "__main__":
    try:
        workloads = load_workloads_from_folder(workload_folder)
        if not workloads:
            print("No valid JSON workloads found in the specified folder.")
        else:
            asyncio.run(automated_batch_profiling(workloads))
    except KeyboardInterrupt:
        print("Process stopped")
