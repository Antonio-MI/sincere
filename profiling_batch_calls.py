import aiohttp
import asyncio
import time
import numpy as np
import json
import os
import random

# PARAMS TO GIVE FROM THE SH SCRIPT
# BATCH SIZES, MODELS TO PROFILE, NUM RUNS PER BATCH SIZE

# Seed for reproducible trials
np.random.seed(42)
random.seed(42)

# Define the API endpoint
api_url = "http://127.0.0.1:5000/inference"

# Folder where workload jsons are located
workload_folder = "./workloads"

# List of batch sizes to profile
# Full list of powers of 2 = primes
# batch_sizes = [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 17, 19, 23, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64, 67, 71, 73, 79, 83, 89, 
#                97, 101, 103, 107, 109, 113, 127, 128, 131, 137, 139, 144, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
#                199, 211, 223, 227, 229, 233, 239, 241, 251, 256, 257, 263, 269, 271, 277, 281, 283,	293, 307, 311, 313, 317, 331,
#                337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
#                467, 479, 487, 491, 499, 503, 509, 512, 1024, 2048] #, 4096, 8192, 16384, 32768] 
batch_sizes = [1, 2, 3, 4, 5, 7, 8, 11, 13, 16, 19, 23, 29, 32, 41, 53, 64, 79, 83, 89, 109, 128, 149, 167, 191, 211, 233, 256, 307, 353, 409, 457, 512, 1024, 2048]

# List of models to profile
models_to_profile = ["granite-7b", "gemma-7b", "llama3-8b"] #["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m"]  # Add more models as needed

# Number of profiling runs per batch size
num_runs_per_batch_size = 10

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
                if "error" in result:
                    print(f"Error: {result['error']}")
                    if "Out of memory" in result["error"]:
                        # Return a specific message to indicate OOM error
                        return "OOM_ERROR"
                print(f"Response: {result}")
            elif response.status == 500:  # Handle server errors like OOM
                result = await response.json()
                if "error" in result:
                    print(f"Server Error: {result['error']}")
                    if "Out of memory" in result["error"]:
                        return "OOM_ERROR"
            else:
                print(f"Error: {response.status} - {await response.text()}")
    except Exception as e:
        print(f"Failed to make the request: {e}")

async def profile_batch_size(session, model, batch_size, workloads):
    tasks = []
    for _ in range(batch_size):
        # Select a random workload from the list
        workload = np.random.choice(workloads)
        workload['model_alias'] = model  # Set the current model alias
        workload['batch_size'] = batch_size  # Add the batch size to the workload

        # Send the request asynchronously
        tasks.append(asyncio.create_task(send_request(session, workload)))

    # Wait for all tasks in this batch size to complete
    results = await asyncio.gather(*tasks)

    # Check if any task returned an "OOM_ERROR"
    if "OOM_ERROR" in results:
        print(f"Out of memory error occurred for model {model} with batch size {batch_size}. Skipping further runs for this model.")
        return "OOM_ERROR"

    return "SUCCESS"

async def automated_batch_profiling(workloads):
    async with aiohttp.ClientSession() as session:
        for model in models_to_profile:
            print(f"Profiling model: {model}")
            model_has_error = False  # Flag to check if an OOM error occurred for the model

            for batch_size in batch_sizes:
                if model_has_error:  # Skip to the next model if OOM error occurred
                    break

                print(f"  Profiling batch size: {batch_size}")
                for run in range(num_runs_per_batch_size):
                    print(f"    Run {run + 1} for batch size {batch_size}")
                    result = await profile_batch_size(session, model, batch_size, workloads)

                    if result == "OOM_ERROR":
                        # Stop further profiling for this model and move to the next model
                        model_has_error = True  # Set flag to skip further batch sizes
                        break

                    await asyncio.sleep(1)  # Small delay between runs

if __name__ == "__main__":
    try:
        workloads = load_workloads_from_folder(workload_folder)
        if not workloads:
            print("No valid JSON workloads found in the specified folder.")
        else:
            asyncio.run(automated_batch_profiling(workloads))
    except KeyboardInterrupt:
        print("Process stopped")
