import threading
import time
import uuid
import os
from queue import Queue
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import pandas as pd

# Folder containing models
base_dir = "./models"

# Select device, cpu for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) # Check with nvidia-smi

# To save current model loaded name and model, and its tokenizer
loaded_models = {}
batch_timers = {}

# # Dictionaries to store mean load and unload times
# model_load_times = {}
# model_unload_times = {}

# Load model profiling data
model_profiling = pd.read_csv("./outputs/model_loading_times_cpu_20240903_120841.csv")  # Specify the correct path to your CSV file

# Create dictionaries to store loading and unloading times
model_load_times = model_profiling.set_index("model_name")["mean_loading_time"].to_dict()
model_unload_times = model_profiling.set_index("model_name")["mean_unloading_time"].to_dict()


# Queues for incoming and running requests
incoming_request_batches = {}
running_request_batches = {}

# Manually set batch size for now
default_batch_size = 4
# Time constraint for batch processing
batch_time_limit = 5  # Seconds

# List of allowed models
allowed_models = ["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m"]



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
            if device=="cuda": 
                torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        print(f"Unloaded previous model")

    model_dir = os.path.join(base_dir, model_alias)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Checfk if this padding works for every model
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

def process_batch(model_alias, condition, batch_size):
    global incoming_request_batches, running_request_batches, batch_timers

    print(f"{condition} condition met for model {model_alias}")

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

        # Perform inference using the pipeline
        pipe = pipeline(
            "text-generation",
            model=loaded_models[model_alias]["model"],
            tokenizer=loaded_models[model_alias]["tokenizer"],
            device=device,  
        )

        start_time = time.perf_counter()
        responses = {}
        for i, output in enumerate(pipe(batch_generator, max_new_tokens=32, batch_size=batch_size)):
            try:
                generated_text = output[0]['generated_text']
                request_id = batch[i]['id']
                responses[request_id] = generated_text
                #print(f"Processed request ID {request_id} with model {model_alias}")
            except IndexError:
                print(f"IndexError: Output index {i} is out of range.")
                continue  # Skip this entry if an error occurs
            except Exception as e:
                print(f"Error processing response: {e}")
                continue  # Handle unexpected errors gracefully

        end_time = time.perf_counter()
        print(f"Processed batch: {list(responses.keys())} with model {model_alias} in {end_time - start_time:.4f} seconds")

        # Reset the timer for the next batch
        batch_timers[model_alias] = None

        # Return a list of completed inference IDs (for debugging purposes)
        return list(responses.keys())

def background_batch_processor():
    while True:
        current_time = time.time()
        for model_alias, timer in list(batch_timers.items()):
            # if timer is not None and (current_time - timer) >= batch_time_limit:
            if timer is not None and current_time >= timer:
                if model_alias in incoming_request_batches and not incoming_request_batches[model_alias].empty():
                    batch_size = incoming_request_batches[model_alias].qsize()
                    #print(f"Moving batch for {model_alias} from incoming to running due to time limit")
                    running_request_batches[model_alias] = Queue()
                    while not incoming_request_batches[model_alias].empty():
                        running_request_batches[model_alias].put(incoming_request_batches[model_alias].get())
                    #print(f"Triggering batch processing for model {model_alias}")
                    process_batch(model_alias, "Time limit", batch_size)
        time.sleep(0.1)  # Change if requests arrive in the order of miliseconds

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    global batch_timers, incoming_request_batches, running_request_batches

    model_alias = request.json.get('model_alias')
    prompt = request.json.get('prompt')
    request_id = str(uuid.uuid4())[:8]  # Generate a unique request ID
    print(f"Request with ID {request_id} for model {model_alias} received")
    
    # Check if the model is in the allowed models list
    if model_alias not in allowed_models:
        return jsonify({
            'error': f"Model '{model_alias}' is not allowed."
        }), 400  # Return a 400 Bad Request response


    # Initialize the request batch and timer for this model if not already done
    if model_alias not in incoming_request_batches:
        incoming_request_batches[model_alias] = Queue()
        running_request_batches[model_alias] = Queue()
        batch_timers[model_alias] = None

    # Store the request data for batching
    request_data = {
        'id': request_id,
        'model_alias': model_alias,
        'prompt': prompt
    }
    
    incoming_request_batches[model_alias].put(request_data)

    batch_size = default_batch_size

    # Start the timer if this is the first request in the batch
    # if batch_timers[model_alias] is None:
    #     batch_timers[model_alias] = time.time()

    # Adjust time limit based on model loading/unloading times
    current_loaded_model = list(loaded_models.keys())[0] if loaded_models else None
    loading_time = model_load_times.get(model_alias, 0)
    unloading_time = model_unload_times.get(current_loaded_model, 0) if current_loaded_model else 0

    adjusted_time_limit = batch_time_limit - loading_time - unloading_time
    adjusted_time_limit = max(adjusted_time_limit, 0)  # Ensure it's not negative

    # Start the timer if this is the first request in the batch
    if batch_timers[model_alias] is None:
        batch_timers[model_alias] = time.time() + adjusted_time_limit  # Adjust the timer

    # Check if batch size is met
    if incoming_request_batches[model_alias].qsize() >= batch_size:
        print(f"Moving batch for {model_alias} from incoming to running due to batch size")
        running_request_batches[model_alias] = Queue()
        while not incoming_request_batches[model_alias].empty():
            running_request_batches[model_alias].put(incoming_request_batches[model_alias].get())
        # Process the batch because the batch size was met
        completed_inference_ids = process_batch(model_alias, "Batch size", batch_size)
        # return jsonify({
        #     'message': f'Inferences completed with {model_alias}': completed_inference_ids
        # })
        return jsonify({
        'message': f"f'Inferences completed with {model_alias}: {completed_inference_ids}'"
    })
    return jsonify({
        'message': f"Request queued with ID {request_id} for model {model_alias}"
    })


def get_model_size_in_bytes(model):
    # Calculate the size of the model parameters
    param_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4  # 4 bytes per float32 parameter
    return param_size

def get_tokenizer_size_in_bytes(tokenizer_dir):
    # Calculate the size of all files in the tokenizer directory
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(tokenizer_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


if __name__ == '__main__':

    # Model profiling can be a separate script and this one would read the csv

    # Directory for models
    # os.makedirs("./outputs", exist_ok=True)
    # results = []

    # # Iterate through all the models available to profile their mean load and unload
    # for model in ["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m"]:
    #     load_times = []
    #     unload_times = []
    #     model_sizes = []

    #     for _ in range(10):
    #         # Profile the loading time
    #         load_start_time = time.time()
    #         model_dir = os.path.join(base_dir, model)
    #         tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #         model_instance = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    #         load_time = time.time() - load_start_time
    #         load_times.append(load_time)

    #         # Calculate model size (parameters + tokenizer)
    #         model_param_size = get_model_size_in_bytes(model_instance)
    #         tokenizer_size = get_tokenizer_size_in_bytes(model_dir)
    #         total_model_size = model_param_size + tokenizer_size
    #         model_sizes.append(total_model_size)

    #         # Profile the unloading time
    #         unload_start_time = time.time()
    #         del model_instance
    #         del tokenizer
    #         if device=="cuda": 
    #             torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
    #         unload_time = time.time() - unload_start_time
    #         unload_times.append(unload_time)

    #     # Calculate the mean load and unload times
    #     mean_load_time = round(np.mean(load_times), 4)
    #     mean_unload_time = round(np.mean(unload_times), 4)
    #     model_size = round(np.mean(model_sizes) / (1024**3), 2)  # Convert bytes to GB

    #     results.append((
    #                         model, model_size, mean_load_time, mean_unload_time
    #                     ))

    #     # Store the mean times in the dictionaries
    #     model_load_times[model] = mean_load_time
    #     model_unload_times[model] = mean_unload_time

    #     print(f"Profiled model {model} - Load time: {mean_load_time:.4f}s, Unload time: {mean_unload_time:.4f}s")

    # # Save model profiling info into a csv
    # df = pd.DataFrame(
    #             results,
    #             columns=[
    #                 "model_name", "model_size", "mean_loading_time", "mean_unloading_time"
    #             ],
    #             )

    # timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # csv_filename = f"model_loading_times_{device}_{timestamp}.csv"
    # csv_path = "outputs/" + csv_filename
    # file_exists = os.path.isfile(csv_path)
    # if file_exists:
    #     df.to_csv(csv_path, mode="a", header=False, index=False)
    # else:
    #     df.to_csv(csv_path, index=False)

    # Start the background thread to process batches based on time limit
    threading.Thread(target=background_batch_processor, daemon=True).start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
