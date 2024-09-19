import uuid
import os
from queue import Queue
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import pandas as pd
import platform
import logging
from monitor import Monitor
import threading

# PARAMS TO RECEIVE FROM THE SH FILE
# LIST OF ALLOWED MODELS (SHOULD BE EQUAL TO THE LIST OF MODELS TO PROFILE IN THE CALLS)

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# Folder containing models
base_dir = "./models"

# Ensure that the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(filename=f"logs/batch_processing_debug_{timestamp}.log", level=logging.DEBUG, format="%(asctime)s %(message)s")

# Select device, cpu for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}") # Check with nvidia-smi

# Save machine name to identify csv
machine_name = platform.node()

# To save current model loaded name and model, and its tokenizer
loaded_models = {}

# Queues for incoming and running requests
incoming_request_batches = {}
running_request_batches = {}

# List of allowed models
allowed_models = ["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m", "granite-7b", "gemma-7b", "llama3-8b"]

# Lock to not process another batch until the current one has finished
batch_processing_lock = threading.Lock()

# Initialize the GPU monitoring
monitoring = False
if device.type == "cuda":
    monitoring = True
    logging.debug(f"Monitoring status set to {monitoring}")
    monitor = Monitor(cuda_enabled=True)


# Function to load models
def load_model(model_alias):
    global loaded_models

    if model_alias in loaded_models:
        print(f"Model {model_alias} already loaded")
        return

    # Unload the previous model
    if loaded_models:
        for old_model_alias in list(loaded_models.keys()):
            del loaded_models[old_model_alias]
            if device.type == "cuda":
                torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        print(f"Unloaded previous model")

    model_dir = os.path.join(base_dir, model_alias)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "left"  # Set padding to the left side for decoder-only architectures
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loaded model {model_alias}")

    loaded_models[model_alias] = {"model": model, "tokenizer": tokenizer}


# Function to generate a dataset for batching
def create_batch_generator(batch):
    for request_data in batch:
        yield request_data['prompt']


def process_batch(model_alias, batch_size):
    global incoming_request_batches, running_request_batches, monitoring, monitor

    with batch_processing_lock:

        if running_request_batches.get(model_alias):
            batch = list(running_request_batches[model_alias].queue)
            running_request_batches[model_alias].queue.clear()  # Clear the running queue after copying
            if not batch:
                print(f"No batch to process for model {model_alias}")
                return

            print(f"Loading model {model_alias}")
            load_model(model_alias)

            # Create a generator for batching
            batch_generator = create_batch_generator(batch)

            # Save current batch size
            current_batch_size = len(batch)

            try:
                # Perform inference using the pipeline
                pipe = pipeline(
                    "text-generation",
                    model=loaded_models[model_alias]["model"],
                    tokenizer=loaded_models[model_alias]["tokenizer"],
                    device=device,
                    torch_dtype=torch.float16
                )

                start_time = time.perf_counter()
                responses = {}
                for i, output in enumerate(pipe(batch_generator, max_new_tokens=128, batch_size=batch_size)):
                    try:
                        generated_text = output[0]['generated_text']
                        request_id = batch[i]['id']
                        responses[request_id] = generated_text
                    except IndexError:
                        print(f"IndexError: Output index {i} is out of range.")
                        continue  # Skip this entry if an error occurs
                    except Exception as e:
                        print(f"Error processing response: {e}")
                        continue  # Handle unexpected errors gracefully

                end_time = time.perf_counter()
                
                print(f"Processed batch: {list(responses.keys())} with model {model_alias}")

                if monitoring == True:
                    logging.debug("Saving sys info")
                    sys_info = monitor.get_sys_info()

                batch_inference_time = round(end_time - start_time,3)

                # Calculate latency for each request
                for request in batch:
                    request_id = request['id']
                    request_time = request['request_time']
                    latency = round(end_time - request_time,3)  # Time since the request was received until the batch was processed
                    logging.debug(f"Latency for request {request_id} with model {model_alias}: {latency:.4f} seconds")

                    # Save the latency result to a CSV file
                    if monitoring == True:
                        logging.debug("Saving results with gpu monitoring")
                        save_measurements_and_monitor(request_id, request["arrival_time"], model_alias, current_batch_size, latency, batch_inference_time, sys_info)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of GPU memory. Error: {e}")
                    torch.cuda.empty_cache()  # Clear GPU memory
                    print(f"GPU memory cleared after OOM error.")
                    latency = "None"
                    batch_inference_time = "None"
                    sys_info = "None"
                    save_measurements_and_monitor(request_id, request_time, model_alias, current_batch_size, latency, batch_inference_time, sys_info)
                    return None, f"Out of memory error while processing batch for {model_alias}"
                else:
                    print(f"Unexpected runtime error: {e}")
                    torch.cuda.empty_cache()
                    #elapsed_time = "None"
                    latency = "None"
                    batch_inference_time = "None"
                    sys_info = "None"
                    save_measurements_and_monitor(request_id, request_time, model_alias, current_batch_size, latency, batch_inference_time, sys_info)
                    return None, f"Unexpected error while processing batch for {model_alias}"

        # Clean cache after processing
        torch.cuda.empty_cache()

        return list(responses.keys()), None

def save_measurements_and_monitor(request_id, request_time, model_alias, batch_size, latency, batch_inference_time, sys_info):
    csv_filename = f"batch_profiling_results_{machine_name}_{device}_{timestamp}.csv"
    csv_path = os.path.join("outputs", csv_filename)
    data = {
        "model": model_alias,
        "batch_size": batch_size,
        "latency (s)": latency,
        "processing time (s)": batch_inference_time,
        "throughput (qps)": "None" if batch_inference_time=="None" else round(batch_size/batch_inference_time, 2)
    }

    # Update the data dictionary with the sys_info entries
    data.update(sys_info)
    
    # Convert the combined data into a DataFrame
    df = pd.DataFrame([data])

    file_exists = os.path.isfile(csv_path)
    if file_exists:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def save_profiling_result(model_alias, batch_size, processing_time):
    csv_filename = f"batch_profiling_results_{machine_name}_{device}_{timestamp}.csv"
    csv_path = os.path.join("outputs", csv_filename)
    data = {
        "model_alias": model_alias,
        "batch_size": batch_size,
        "processing_time": processing_time,  
    }
    df = pd.DataFrame([data])
    file_exists = os.path.isfile(csv_path)
    if file_exists:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    try:
        global incoming_request_batches, running_request_batches

        model_alias = request.json.get('model_alias')
        prompt = request.json.get('prompt')
        batch_size = int(request.json.get('batch_size'))  # Get batch size from the request
        request_id = str(uuid.uuid4())[:8]  # Generate a unique request ID
        request_time = time.perf_counter()
        arrival_time = time.localtime()
        print(f"Request with ID {request_id} for model {model_alias} and batch size {batch_size} received")

        # Check if the model is in the allowed models list
        if model_alias not in allowed_models:
            return jsonify({
                'error': f"Model '{model_alias}' is not allowed."
            }), 400  # Return a 400 Bad Request response

        # Initialize the request batch for this model if not already done
        if model_alias not in incoming_request_batches:
            incoming_request_batches[model_alias] = Queue()
            running_request_batches[model_alias] = Queue()

        # Store the request data for batching
        request_data = {
            'id': request_id,
            'model_alias': model_alias,
            'prompt': prompt,
            'request_time': request_time,
            'arrival_time': arrival_time
        }

        incoming_request_batches[model_alias].put(request_data)

        # Check if batch size is met
        if incoming_request_batches[model_alias].qsize() >= batch_size:
            print(f"Moving batch for {model_alias} from incoming to running due to batch size")
            running_request_batches[model_alias] = Queue()
            while not incoming_request_batches[model_alias].empty():
                running_request_batches[model_alias].put(incoming_request_batches[model_alias].get())
            # Process the batch because the batch size was met
            completed_inference_ids, error = process_batch(model_alias, batch_size)

            # If there's an error (e.g., OOM), return it in the response
            if completed_inference_ids is None:
                return jsonify({'error': error}), 500

            return jsonify({
                'message': f"Inferences completed with {model_alias}: {completed_inference_ids}"
            })

        return jsonify({
            'message': f"Request queued with ID {request_id} for model {model_alias} and batch size {batch_size}"
        })

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Out of GPU memory. Error: {e}")
            return jsonify({'error': f"Out of memory error while processing batch for {model_alias}"}), 500
        else:
            print(f"Runtime error: {e}")
            return jsonify({'error': f"Unexpected error: {e}"}), 500


@app.route('/health')
def health():
    return 'OK', 200


if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000)
