from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import time
import numpy as np
import threading

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

# Queue to store incoming requests
request_queue = []
batch_size = 2  # Fixed batch size
batch_timeout = 5  # Timeout to process smaller batches
batch_lock = threading.Lock()

# Function to load models
def load_model(model_alias):
    global current_model_alias, loaded_model, loaded_tokenizer

    if model_alias == current_model_alias:
        print("Model already loaded")
        return

    if loaded_model is not None:
        del loaded_model
        del loaded_tokenizer
        torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        print(f"Unloaded model {current_model_alias}")

    model_dir = os.path.join(base_dir, model_alias)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    print(f"Loaded model {model_alias}")

    current_model_alias = model_alias
    loaded_model = model
    loaded_tokenizer = tokenizer

def process_batch():
    global request_queue
    while True:
        with batch_lock:
            batch = []  # Initialize batch as an empty list

            if len(request_queue) >= batch_size:
                # Process a full batch
                batch = request_queue[:batch_size]
                request_queue = request_queue[batch_size:]
            elif len(request_queue) > 0:
                # Wait for the batch timeout before processing smaller batches
                time.sleep(batch_timeout)
                if len(request_queue) > 0:
                    batch = request_queue
                    request_queue = []

        if batch:
            model_alias = batch[0]['model_alias']
            prompts = [req['prompt'] for req in batch]

            load_model(model_alias)
            pipe = pipeline(
                "text-generation",
                model=loaded_model,
                tokenizer=loaded_tokenizer,
                device=0 if torch.cuda.is_available() else -1,  # GPU (0) or CPU (-1)
            )

            outputs = pipe(prompts, max_new_tokens=128)

            # Loop through the outputs and assign them to the corresponding requests
            for i, output in enumerate(outputs):
                # Access the generated text from the dictionary
                generated_text = output['generated_text']
                batch[i]['response'] = generated_text

            # Set each request's event immediately after processing it
            for req in batch:
                req['event'].set()



app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    model_alias = request.json.get('model_alias')
    prompt = request.json.get('prompt', '')

    event = threading.Event()
    request_data = {
        "model_alias": model_alias,
        "prompt": prompt,
        "event": event,
        "response": None
    }

    with batch_lock:
        request_queue.append(request_data)

    # Wait until the event is set, with a timeout to prevent indefinite blocking
    event_is_set = event.wait(timeout=30)

    if not event_is_set:
        return jsonify({"error": "Request timed out"}), 504

    return jsonify({'response': request_data['response']})

def profile_models():
    for model in ["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m"]:
        load_times = []
        unload_times = []

        for _ in range(10):
            load_start_time = time.time()
            model_dir = os.path.join(base_dir, model)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model_instance = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
            load_time = time.time() - load_start_time
            load_times.append(load_time)

            unload_start_time = time.time()
            del model_instance
            del tokenizer
            #torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
            unload_time = time.time() - unload_start_time
            unload_times.append(unload_time)

        mean_load_time = np.mean(load_times)
        mean_unload_time = np.mean(unload_times)

        model_load_times[model] = mean_load_time
        model_unload_times[model] = mean_unload_time

        print(f"Profiled model {model} - Load time: {mean_load_time:.4f}s, Unload time: {mean_unload_time:.4f}s")

if __name__ == '__main__':
    # Run profiling 
    profile_models()

    # Start the batch processing thread
    threading.Thread(target=process_batch, daemon=True).start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)


