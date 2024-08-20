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
current_model_alias = None
loaded_model = None
loaded_tokenizer = None

# Dictionaries to store mean load and unload times
model_load_times = {}
model_unload_times = {}

# Lists to store requests and their IDs for batching
request_batch = []
#batch_size = random.choice([4])  # Fixed batch size (for now)

# Time constraint variables
batch_start_time = None
batch_time_limit = 5  # Time limit in seconds for batch processing

# Function to load models
def load_model(model_alias):
    global current_model_alias, loaded_model, loaded_tokenizer

    if model_alias == current_model_alias:
        print("Model already loaded")
        return

    if loaded_model is not None:
        del loaded_model
        del loaded_tokenizer
        #torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        print(f"Unloaded model {current_model_alias}")

    model_dir = os.path.join(base_dir, model_alias)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Checfk if this padding works for every model
    tokenizer.padding_side = "left"  # Set padding to the left side for decoder-only architectures
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    
    # Set the pad_token_id to the eos_token_id if padding is not defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loaded model {model_alias}")

    current_model_alias = model_alias
    loaded_model = model
    loaded_tokenizer = tokenizer

# Function to generate a dataset for batching
def create_batch_generator(batch):
    for request_data in batch:
        yield request_data['prompt']

app = Flask(__name__)

def process_batch(condition, batch_size):
    global request_batch, batch_start_time

    print(f"{condition} condition met")

    if request_batch:
        # Perform batch inference
        model_alias = request_batch[0]['model_alias']
        load_model(model_alias)

        # Create a generator or dataset for batching
        batch_generator = create_batch_generator(request_batch)

        # Perform inference using the pipeline
        pipe = pipeline(
            "text-generation",
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device="cpu",  
        )

        # for out in tqdm(pipe(batch_generator, max_new_tokens=32, batch_size=batch_size )):
        #     pass

        outputs = pipe(batch_generator, max_new_tokens=32, batch_size=batch_size)

        # Prepare the responses
        responses = {}
        for i, output in enumerate(outputs):
            # Since output is a list containing one dictionary, access the dictionary first
            generated_text = output[0]['generated_text']
            request_id = request_batch[i]['id']
            responses[request_id] = generated_text

        # Clear the batch after processing
        request_batch.clear()

        # Reset the timer for the next batch
        batch_start_time = None

        print(f"Processed batch and cleared requests: {list(responses.keys())}")

        # Return a list of completed inference IDs (for debugging purposes)
        return list(responses.keys())

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    global batch_start_time

    model_alias = request.json.get('model_alias')
    prompt = request.json.get('prompt')
    request_id = str(uuid.uuid4())  # Generate a unique request ID
    print(f"Request with ID {request_id} received")
    
    # Store the request data for batching
    request_data = {
        'id': request_id,
        'model_alias': model_alias,
        'prompt': prompt
    }
    request_batch.append(request_data)

    batch_size = 4

    # Start the timer if this is the first request in the batch
    if batch_start_time is None:
        batch_start_time = time.time()

    # Check if batch size is met or if time constraint is met
    if len(request_batch) >= batch_size:
        # Process the batch because the batch size was met
        completed_inference_ids = process_batch("Batch size", batch_size)
        # Return the response to the original request
        return jsonify({
            'Inferences completed': completed_inference_ids
        })

    elif (time.time() - batch_start_time) >= batch_time_limit:
        # Process the batch because the time limit was met
        batch_size = len(request_batch)
        completed_inference_ids = process_batch("Time limit", batch_size)
        # Return the response to the original request
        return jsonify({
            'Inferences completed': completed_inference_ids
        })

    return jsonify({
        'message': f"Request queued with ID {request_id}"
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
