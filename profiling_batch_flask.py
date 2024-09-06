import uuid
import os
from queue import Queue
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import pandas as pd
import platform

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# Folder containing models
base_dir = "./models"

# Select device, cpu for now
device = "cuda"
print(device)  # Check with nvidia-smi

# Save machine name to identify csv
machine_name = platform.node()

# To save current model loaded name and model, and its tokenizer
loaded_models = {}

# Queues for incoming and running requests
incoming_request_batches = {}
running_request_batches = {}

# List of allowed models
allowed_models = ["granite-7b", "gemma-7b", "llama3-8b"]

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
            if device == "cuda":
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
    global incoming_request_batches, running_request_batches

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

        try:
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
                except IndexError:
                    print(f"IndexError: Output index {i} is out of range.")
                    continue  # Skip this entry if an error occurs
                except Exception as e:
                    print(f"Error processing response: {e}")
                    continue  # Handle unexpected errors gracefully

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"Processed batch: {list(responses.keys())} with model {model_alias} in {elapsed_time:.4f} seconds")

            # Save the result to a CSV file
            save_profiling_result(model_alias, batch_size, elapsed_time)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Out of GPU memory. Error: {e}")
                torch.cuda.empty_cache()  # Clear GPU memory
                print(f"GPU memory cleared after OOM error.")
                elapsed_time = "None"
                save_profiling_result(model_alias, batch_size, elapsed_time)
                return None, f"Out of memory error while processing batch for {model_alias}"
            else:
                print(f"Unexpected runtime error: {e}")
                elapsed_time = "None"
                save_profiling_result(model_alias, batch_size, elapsed_time)
                return None, f"Unexpected error while processing batch for {model_alias}"


        # Clean cache after processing
        torch.cuda.empty_cache()

        return list(responses.keys()), None

def save_profiling_result(model_alias, batch_size, processing_time):
    csv_filename = f"batch_profiling_results_{machine_name}_{device}_{timestamp}.csv"
    csv_path = os.path.join("outputs", csv_filename)
    data = {
        "model_alias": model_alias,
        "batch_size": batch_size,
        "processing_time": processing_time,
        "throughput": "None" if processing_time=="None" else round(batch_size/processing_time, 2)
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
            'prompt': prompt
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

if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000)
