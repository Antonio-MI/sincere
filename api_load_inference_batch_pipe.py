from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import time
import uuid  # To generate unique request IDs
import numpy as np
import random
#from tqdm.auto import tqdm

# Folder containing models
base_dir = "./models"

# Select device, cpu for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# To save current model loaded name and model, and its tokenizer
loaded_models = {}
batch_timers = {}

# Dictionaries to store mean load and unload times
model_load_times = {}
model_unload_times = {}

# Lists to store requests and their IDs for batching
request_batches = {}

default_batch_size = 4
#batch_size = random.choice([4])  # Fixed batch size (for now)

# Time constraint
batch_time_limit = 5  # Time limit in seconds for batch processing

# Function to load models
def load_model(model_alias):
    global loaded_models

    if model_alias in loaded_models:
        print("Model already loaded")
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
    
    # Set the pad_token_id to the eos_token_id if padding is not defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loaded model {model_alias}")

    loaded_models[model_alias] = {"model": model, "tokenizer": tokenizer}


# Function to generate a dataset for batching
def create_batch_generator(batch):
    for request_data in batch:
        yield request_data['prompt']

app = Flask(__name__)

def process_batch(model_alias, condition, batch_size):
    global request_batches, batch_start_time

    print(f"{condition} condition met")

    if request_batches[model_alias]:
        
        load_model(model_alias)

        # Create a generator for batching
        batch_generator = create_batch_generator(request_batches[model_alias])

        # Perform inference using the pipeline
        pipe = pipeline(
            "text-generation",
            model=loaded_models[model_alias]["model"],
            tokenizer=loaded_models[model_alias]["tokenizer"],
            device="cpu",  
        )

        # for out in tqdm(pipe(batch_generator, max_new_tokens=32, batch_size=batch_size )):
        #     pass

        # Measure the inference time
        start_time = time.perf_counter()
        # Check if max_new_tokens affects the inference time - It seems that it doesnt
        #outputs = pipe(batch_generator, max_new_tokens=256, batch_size=batch_size)
        end_time = time.perf_counter()

        # Process outputs iteratively without checking for length
        responses = {}
        for i, output in enumerate(pipe(batch_generator, max_new_tokens=32, batch_size=batch_size)):
            try:
                # Process the generated text for each output
                generated_text = output[0]['generated_text']
                request_id = request_batches[model_alias][i]['id']
                responses[request_id] = generated_text
            except IndexError:
                print(f"IndexError: Output index {i} is out of range.")
                continue  # Skip this entry if an error occurs
            except Exception as e:
                print(f"Error processing response: {e}")
                continue  # Handle unexpected errors gracefully

        # # Calculate the total and average inference time
        # total_inference_time = end_time - start_time
        # average_inference_time = total_inference_time / batch_size
        # print(f"Total inference time for batch: {total_inference_time:.6f} seconds")
        # print(f"Average inference time per request: {average_inference_time:.6f} seconds")

        # # Prepare the responses
        # responses = {}
        # for i, output in enumerate(outputs):
        #     generated_text = output[0]['generated_text']
        #     request_id = request_batches[model_alias][i]['id']
        #     responses[request_id] = generated_text

        # Clear the batch after processing
        request_batches[model_alias].clear()

        # Reset the timer for the next batch
        batch_timers[model_alias] = None

        print(f"Processed batch: {list(responses.keys())}")

        # Return a list of completed inference IDs (for debugging purposes)
        return list(responses.keys())

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    global batch_timers, request_batches

    model_alias = request.json.get('model_alias')
    prompt = request.json.get('prompt')
    request_id = str(uuid.uuid4())[:8]  # Generate a unique request ID
    print(f"Request with ID {request_id} for model {model_alias} received")
    
    # Initialize the request batch and timer for this model if not already done
    if model_alias not in request_batches:
        request_batches[model_alias] = []
        batch_timers[model_alias] = None

    # Store the request data for batching
    request_data = {
        'id': request_id,
        'model_alias': model_alias,
        'prompt': prompt
    }
    request_batches[model_alias].append(request_data)

    batch_size = default_batch_size

    # Start the timer if this is the first request in the batch
    if batch_timers[model_alias] is None:
        batch_timers[model_alias]= time.time()

    # Check if batch size is met or if time constraint is met
    if len(request_batches[model_alias]) >= batch_size:
        # Process the batch because the batch size was met
        completed_inference_ids = process_batch(model_alias, "Batch size", batch_size)
        # Return the response to the original request
        return jsonify({
            f'Inferences completed with {model_alias}': completed_inference_ids
        })

    elif (time.time() - batch_timers[model_alias]) >= batch_time_limit:
        # Process the batch because the time limit was met
        batch_size = len(request_batches[model_alias])
        completed_inference_ids = process_batch(model_alias, "Time limit", batch_size)
        # Return the response to the original request
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

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
