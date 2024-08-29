import requests
import time
import numpy as np
import json
import random
import os
import aiohttp
import asyncio


# Seed for reproducible trials
random.seed(42)
np.random.seed(42)

# Define the API endpoint
api_url = "http://127.0.0.1:5000/inference"

# Folder where workload jsons are located
workload_folder = "./workloads"

# Define the parameters for the Gamma distribution
shape, scale = 1.0, 1.0  # Shape (k) and scale (θ) for the Gamma distribution

run_duration = 20 #seconds

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
                    filtered_payload= {
                        "model_alias": data["model_alias"],
                        "prompt": data["prompt"]
                    }
                    workloads.append(filtered_payload)
                else:
                    print(f"Missing requiered fields in {file_path}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {file_path}")
    return workloads

def make_api_call(workload):
    try:
        #print("Sending request")
        response = requests.post(api_url, json=workload)
        #print("Response received")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to make the request: {e}")


def automated_calls(workloads, run_duration):
    start_time = time.time()
    #print("Timer started")
    while time.time() - start_time < run_duration:
        # Select a random workload from the list
        workload = np.random.choice(workloads)
        
        # Make the API call
        # print("Proceding to call API")
        make_api_call(workload)
        # print("Api called")
        # Generate a sleep interval following the Gamma distribution
        interval = np.random.gamma(shape, scale)
        #print(interval)

        #Miliseconds
        # print(f"Sleeping for {interval:.2f} miliseconds")
        # # Sleep for the generated interval before the next call
        # time.sleep(interval/1000)

        #Seconds
        print(f"Sleeping for {interval:.2f} seconds")
        # Sleep for the generated interval before the next call
        time.sleep(interval)


if __name__ == "__main__":
    try:
        workloads = load_workloads_from_folder(workload_folder)
        # print("Workloads loaded")
        if not workloads:
            print("No valid JSON workloads found in the specified folder.")
        else:
            # print("Proceding to make calls")
            automated_calls(workloads, run_duration)
            # print("Calls made")

    except KeyboardInterrupt:
        print("Process stopped")