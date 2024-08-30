import threading
import time
import uuid
import os
from queue import Queue
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np

# Folder containing models
base_dir = "./models"

# Select device, cpu for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# To save current model loaded name and model, and its tokenizer
loaded_models = {}
batch_timers = {}

# Dictionaries to store mean load and unload times
model_load_times = {}
model_unload_times = {}

# Queues for incoming and running requests
incoming_request_batches = {}
running_request_batches = {}

# Manually set batch size for now
default_batch_size = 4
# Time constraint for batch processing
batch_time_limit = 5  # Seconds

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
        # torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
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
            if timer is not None and (current_time - timer) >= batch_time_limit:
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
    if batch_timers[model_alias] is None:
        batch_timers[model_alias] = time.time()

    # Check if batch size is met
    if incoming_request_batches[model_alias].qsize() >= batch_size:
        print(f"Moving batch for {model_alias} from incoming to running due to batch size")
        running_request_batches[model_alias] = Queue()
        while not incoming_request_batches[model_alias].empty():
            running_request_batches[model_alias].put(incoming_request_batches[model_alias].get())
        # Process the batch because the batch size was met
        completed_inference_ids = process_batch(model_alias, "Batch size", batch_size)
        return jsonify({
            f'Inferences completed with {model_alias}': completed_inference_ids
        })

    return jsonify({
        'message': f"Request queued with ID {request_id} for model {model_alias}"
    })


if __name__ == '__main__':
    # Iterate through all the models available to profile their mean load and unload
    for model in ["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m"]:
        load_times = []
        unload_times = []

        for _ in range(10):
            # Profile the loading time
            load_start_time = time.time()
            model_dir = os.path.join(base_dir, model)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model_instance = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
            load_time = time.time() - load_start_time
            load_times.append(load_time)

            # Profile the unloading time
            unload_start_time = time.time()
            del model_instance
            del tokenizer
            #torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            unload_time = time.time() - unload_start_time
            unload_times.append(unload_time)

        # Calculate the mean load and unload times
        mean_load_time = np.mean(load_times)
        mean_unload_time = np.mean(unload_times)

        # Store the mean times in the dictionaries
        model_load_times[model] = mean_load_time
        model_unload_times[model] = mean_unload_time

        print(f"Profiled model {model} - Load time: {mean_load_time:.4f}s, Unload time: {mean_unload_time:.4f}s")

    # Start the background thread to process batches based on time limit
    threading.Thread(target=background_batch_processor, daemon=True).start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
