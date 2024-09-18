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
import platform
import logging
from monitor import Monitor


mode = "batchedFCFS" #"FCFS"


# Ensure that the logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
logging.basicConfig(filename=f"logs/batch_processing_debug_{timestamp}.log", level=logging.DEBUG, format="%(asctime)s %(message)s")

# Save machine name to identify csv
machine_name = platform.node()

# Folder containing models
base_dir = "./models"

# Select device, cpu for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}") # Check with nvidia-smi

# To save current model loaded name and model, and its tokenizer
loaded_models = {}

# Queues for incoming and running requests
incoming_request_batches = {}
running_request_batches = {}

# Batch sizes depending on the strategy

if mode == "FCFS":
    logging.debug(f"Scheduling mode set as {mode}")
    allowed_batch_sizes = [1]

if mode == "batchedFCFS":
    logging.debug(f"Scheduling mode set as {mode}")
    allowed_batch_sizes = [16]

# Time constraint for batch processing
# batch_time_limit = 15  # Seconds

# List of allowed models
allowed_models = ["gpt2-124m", "distilgpt2-124m", "gpt2medium-355m", "Stop", "granite-7b", "gemma-7b", "llama3-8b"]

# Lock to not process another batch until the current one has finished
batch_processing_lock = threading.Lock()

# To measure device's time doing inference vs idle
first_request_time = None  # Time when the first request is received
last_request_time = None  # Time when the last batch is processed
total_inference_time = 0  # Global variable to track total inference time

# Flag for log inference % only once
inference_flag = False

# Timer to track SLA of each batch
batch_timers = {}
#padding = True

# Global dictionary to track model usage frequency
# model_usage_count = {}
# total_requests = 0  # Total number of requests processed

# Global dictionary to store flags for each model_alias
# correction_applied = {}

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
        logging.debug(f"Model {model_alias} already loaded")
        return

    # Unload the previous model
    if loaded_models:
        for old_model_alias in list(loaded_models.keys()):
            del loaded_models[old_model_alias]
            if device.type=="cuda": 
                torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        logging.debug(f"Unloaded previous model")

    model_dir = os.path.join(base_dir, model_alias)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Checfk if this padding works for every model
    tokenizer.padding_side = "left"  # Set padding to the left side for decoder-only architectures
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logging.debug(f"Loaded model {model_alias}")

    loaded_models[model_alias] = {"model": model, "tokenizer": tokenizer}

# Function to generate a dataset for batching
def create_batch_generator(batch):
    for request_data in batch:
        yield request_data['prompt']

# def generate_padding_request(model_alias):
#     return {
#         'id': str(uuid.uuid4())[:4], # Shorter ids so it easier to recognize
#         'model_alias': model_alias,
#         'prompt': "This is a padding request to fill the batch.",
#         'request_time': time.perf_counter()
#     }

# def get_processing_time(model_alias, batch_size):
#     """Retrieve processing time for a given model and batch size from profiling data."""
#     return processing_times.get((model_alias, batch_size), batch_time_limit), processing_times_std.get((model_alias, batch_size), batch_time_limit)  # Default to batch_time_limit if not found


def save_measurements(request_id, model_alias, batch_size, latency, batch_inference_time, throughput):
    csv_filename = f"measurements_results_{machine_name}_{device}_{timestamp}.csv"
    csv_path = os.path.join("outputs", csv_filename)
    data = {
        "request_id": request_id,
        "timestamp": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        "model": model_alias,
        "batch_size": batch_size,
        "latency (s)": latency,
        "processing time (s)": batch_inference_time,
        "throughput (qps)": throughput
    }
    df = pd.DataFrame([data])
    file_exists = os.path.isfile(csv_path)
    if file_exists:
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

def save_measurements_and_monitor(request_id, model_alias, batch_size, latency, batch_inference_time, throughput, sys_info):
    csv_filename = f"measurements_results_{machine_name}_{device}_{timestamp}.csv"
    csv_path = os.path.join("outputs", csv_filename)
    data = {
        "request_id": request_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model": model_alias,
        "batch_size": batch_size,
        "latency (s)": latency,
        "processing time (s)": batch_inference_time,
        "throughput (qps)": throughput,
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

# def get_padded_batch_size(current_batch_size, allowed_sizes):
#     for size in allowed_sizes:
#         if size >= current_batch_size:
#             return size
#     return allowed_sizes[-1]


# def get_total_queue_size():
#     """Calculate the total number of requests across all incoming request batches."""
#     total_size = sum(queue.qsize() for queue in incoming_request_batches.values())
#     return total_size


# def adjust_time_limit_based_on_queue_length(base_time_limit, queue_size):
#     """
#     Adjust the batch processing time limit dynamically based on the number of requests in the queue.
#     """
#     # For GPU the correction shouldnt be so severe, since the inference time does not increase linearly
#     if queue_size < 4:
#         # Full time limit
#         return base_time_limit
#     elif 4 <= queue_size < 8:
#         # Reduce time limit by 25%
#         return base_time_limit * 0.6
#     elif 8 <= queue_size < 16:
#         # Reduce time limit by 50%
#         return base_time_limit * 0.5
#     else:
#         # Further reduce or set a minimum time limit if necessary
#         return base_time_limit * 0.25


def process_batch(model_alias, condition, batch_size):
    global incoming_request_batches, running_request_batches, batch_timers, total_inference_time, last_batch_processed_time, total_time, inference_flag, monitoring, monitor

    logging.debug(f"{condition} condition met for model {model_alias}")

    with batch_processing_lock:
        if running_request_batches.get(model_alias):
            batch = list(running_request_batches[model_alias].queue)

            running_request_batches[model_alias].queue.clear()  # Clear the running queue after copying
            if not batch:
                logging.debug(f"No batch to process for model {model_alias}")
                return

            logging.debug(f"Next: call load_model for {model_alias}")
            load_model(model_alias)

            # Create a generator for batching
            batch_generator = create_batch_generator(batch)

            current_batch_size = len(batch)

            # Perform inference using the pipeline
            pipe = pipeline(
                "text-generation",
                model=loaded_models[model_alias]["model"],
                tokenizer=loaded_models[model_alias]["tokenizer"],
                device=device,  
            )

            start_time = time.perf_counter()
            logging.debug(f"Batch processing started for model {model_alias}")

            responses = {}
            for i, output in enumerate(pipe(batch_generator, max_new_tokens=64, batch_size=current_batch_size)):
                try:
                    generated_text = output[0]['generated_text']
                    request_id = batch[i]['id']
                    responses[request_id] = generated_text
                except IndexError:
                    logging.debug(f"IndexError: Output index {i} is out of range.")
                    continue  # Skip this entry if an error occurs
                except Exception as e:
                    logging.debug(f"Error processing response: {e}")
                    continue  # Handle unexpected errors gracefully

            end_time = time.perf_counter()
            logging.debug(f"Processed batch: {list(responses.keys())} with model {model_alias} in {end_time - start_time:.4f} seconds")

            if monitoring == True:
                logging.debug("Saving sys info")
                sys_info = monitor.get_sys_info()

            batch_inference_time = end_time - start_time
            # Add the batch inference time to the total inference time
            total_inference_time += batch_inference_time

            batch_throughput = round(current_batch_size/batch_inference_time, 3)

            # Calculate latency for each request
            for request in batch:
                request_id = request['id']
                request_time = request['request_time']
                latency = end_time - request_time  # Time since the request was received until the batch was processed
                logging.debug(f"Latency for request {request_id} with model {model_alias}: {latency:.4f} seconds")

                # Save the latency result to a CSV file
                if monitoring == True:
                    logging.debug("Saving results with gpu monitoring")
                    save_measurements_and_monitor(request_id, model_alias, current_batch_size, latency, batch_inference_time, batch_throughput, sys_info)
                else:
                    logging.debug("Saving results without gpu monitoring")
                    save_measurements(request_id, model_alias, current_batch_size, latency, batch_inference_time, batch_throughput)

            # Reset the timer for the next batch
            batch_timers[model_alias] = None

            last_batch_processed_time = time.time()

            # After processing the final batch, record the time
            #if all(queue.qsize() == 0 for queue in incoming_request_batches.values()):
                #time.sleep(0.1)
            # if last_batch_processed_time is not None:
            #     if time.time() >= last_batch_processed_time + 5:
            #         if inference_flag==False:

            #             last_batch_processed_time = time.perf_counter()-5

            #             # Log the total time and the percentage of inference time
            #             total_time = last_batch_processed_time - first_request_time
            #             inference_percentage = (total_inference_time / total_time) * 100

            #             logging.debug(f"Total time: {total_time:.4f} seconds")
            #             logging.debug(f"Total inference time: {total_inference_time:.4f} seconds")
            #             logging.debug(f"Inference time as percentage of total time: {inference_percentage:.2f}%")

            #             inference_flag = True

 
        # Return a list of completed inference IDs (for debugging purposes)
        return list(responses.keys())

# def background_batch_processor():
#     global incoming_request_batches, running_request_batches, last_batch_processed_time, total_time, total_inference_time, first_request_time, last_batch_processed_time, total_time, inference_flag
#     while True:
#         current_time = time.time()
#         for model_alias, timer in list(batch_timers.items()):
#             # if timer is not None and (current_time - timer) >= batch_time_limit:
#             if timer is not None and current_time >= timer:
                
#                 #logging.debug(f"Moving batch for {model_alias} from incoming to running due to time limit with batch size {batch_size}")
#                 running_request_batches[model_alias] = Queue()
#                 while not incoming_request_batches[model_alias].empty():
#                     running_request_batches[model_alias].put(incoming_request_batches[model_alias].get())
#                 batch_size = incoming_request_batches[model_alias].qsize()
#                 process_batch(model_alias, "Time limit", batch_size)

#         # Force process any remaining requests that have not met the batch size
#         # for model_alias in list(incoming_request_batches.keys()):  # Iterate over a copy of the keys
#         #     queue = incoming_request_batches[model_alias]
#         #     if not queue.empty():
#         #         running_request_batches[model_alias] = Queue()
#         #         while not incoming_request_batches[model_alias].empty():
#         #             running_request_batches[model_alias].put(incoming_request_batches[model_alias].get())
#         #         batch_size = running_request_batches[model_alias].qsize()
#         #         process_batch(model_alias, "Remaining requests", batch_size)


#         # Check if all requests are processed
#         # if all(queue.qsize() == 0 for queue in incoming_request_batches.values()) and first_request_time is not None:
#         #     time.sleep(0.2)
#         if last_batch_processed_time is not None:
#             if time.time() >= last_batch_processed_time + 5:
#                 if inference_flag==False:

#                     last_batch_processed_time = time.perf_counter()-5

#                 #if inference_flag==False:
#                     #last_batch_processed_time = time.perf_counter()

#                     # Log the total time and the percentage of inference time
#                     total_time = last_batch_processed_time - first_request_time
#                     inference_percentage = (total_inference_time / total_time) * 100

#                     logging.debug(f"Total time: {total_time:.4f} seconds")
#                     logging.debug(f"Total inference time: {total_inference_time:.4f} seconds")
#                     logging.debug(f"Inference time as percentage of total time: {inference_percentage:.2f}%")

#                     inference_flag = True

#         time.sleep(0.1)  # Change if requests arrive in the order of miliseconds

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    global batch_timers, incoming_request_batches, running_request_batches, model_usage_count, total_requests, first_request_time, last_batch_processed_time, total_time, inference_flag

    # Record the time of the first request
    if first_request_time is None:
        first_request_time = time.time()

    model_alias = request.json.get('model_alias')
    prompt = request.json.get('prompt')
    request_id = str(uuid.uuid4())[:8]  # Generate a unique request ID
    request_time = time.perf_counter()  # Time when the request was received
    
    # To finish inference and save time
    remaining_requests = sum(queue.qsize() for queue in running_request_batches.values())
    if model_alias=="Stop" and inference_flag == False:
        while remaining_requests >0:
            remaining_requests = sum(queue.qsize() for queue in running_request_batches.values())
            logging.debug("Waiting for running processes to finish")
            time.sleep(1)
        # Log the total time and the percentage of inference time
        total_time = last_batch_processed_time - first_request_time
        inference_percentage = (total_inference_time / total_time) * 100
        logging.debug(f"Total time: {total_time:.4f} seconds")
        logging.debug(f"Total inference time: {total_inference_time:.4f} seconds")
        logging.debug(f"Inference time as percentage of total time: {inference_percentage:.2f}%")
        inference_flag = True
        logging.debug("END")

        # Return a response indicating that the process is stopping
        return jsonify({"message": "Inference process stopped."}), 200

    logging.debug(f"Request with ID {request_id} for model {model_alias} received")
    
    # Check if the model is in the allowed models list
    if model_alias not in allowed_models:
        return jsonify({
            'error': f"Model '{model_alias}' is not allowed."
        }), 400  # Return a 400 Bad Request response

    # # Update the model usage count
    # total_requests += 1
    # if model_alias in model_usage_count:
    #     model_usage_count[model_alias] += 1
    # else:
    #     model_usage_count[model_alias] = 1

    # # Calculate and print the frequency of this model
    # model_frequency = (model_usage_count[model_alias] / total_requests) * 100
    # print(f"Model {model_alias} has been called {model_usage_count[model_alias]} times ({model_frequency:.2f}% of total requests)")


    # Initialize the request batch and timer for this model if not already done
    if model_alias not in incoming_request_batches:
        incoming_request_batches[model_alias] = Queue()
        running_request_batches[model_alias] = Queue()
        batch_timers[model_alias] = None

    # Store the request data for batching
    request_data = {
        'id': request_id,
        'model_alias': model_alias,
        'prompt': prompt,
        'request_time': request_time
    }
    
    incoming_request_batches[model_alias].put(request_data)

    # # Get the total queue size (across all models)
    # total_queue_size = get_total_queue_size()

    # # Dynamically adjust the batch time limit based on the total queue size
    # adjusted_time_limit = adjust_time_limit_based_on_queue_length(batch_time_limit, total_queue_size)

    # logging.debug(f"Adjusted time limit based on total queue size {total_queue_size}: {adjusted_time_limit:.4f} seconds")


    # batch_size = min(allowed_batch_sizes) #len(incoming_request_batches[model_alias])#max(allowed_batch_sizes)
    # Get the current batch size
    # current_batch_size = incoming_request_batches[model_alias].qsize()


    # # Start the timer if this is the first request in the batch
    # if batch_timers[model_alias] is None:

    #     # Adjust time limit based on model loading/unloading times
    #     current_loaded_model = list(loaded_models.keys())[0] if loaded_models else None
    #     loading_time = model_load_times.get(model_alias, 0)
    #     loading_time_std = model_load_times_std.get(model_alias,0)
    #     unloading_time = model_unload_times.get(current_loaded_model, 0) if current_loaded_model else 0
    #     unloading_time_std = model_unload_times_std.get(current_loaded_model, 0) if current_loaded_model else 0

    #     loading_time = loading_time + 3*loading_time_std
    #     unloading_time = unloading_time + 3*unloading_time_std

    #     processing_times, processing_times_std = get_processing_time(model_alias, min(allowed_batch_sizes))
    #     processing_time = processing_times + 3 * processing_times_std

    #     # Subtract processing, loading, and unloading times from the batch time limit
    #     adjusted_time_limit = batch_time_limit - (loading_time + unloading_time + processing_time) 

    #     adjusted_time_limit = max(adjusted_time_limit, 0)  # Ensure it's not negative

    #     batch_timers[model_alias] = time.time() + adjusted_time_limit  # Adjust the timer

    #     logging.debug(f"Adjusted time limit for model {model_alias}: {adjusted_time_limit:.4f} seconds")


    # expected_batch_size = 8 

    # if batch_timers[model_alias] is not None and current_batch_size > min(allowed_batch_sizes):
    #     processing_times, processing_times_std = get_processing_time(model_alias, expected_batch_size)
    #     processing_time = processing_times + 3 * processing_times_std
    #     # Previuous processing times for smaller batch size, already considered so we substract that
    #     previous_processing_times, previous_processing_times_std = get_processing_time(model_alias, min(allowed_batch_sizes))
    #     previous_processing_time = previous_processing_times + 3 * previous_processing_times_std
    #     batch_timers[model_alias] = batch_timers[model_alias] - (processing_time - previous_processing_time)

    #     # Set the flag to True to indicate correction has been applied
    #     correction_applied[model_alias] = True


    # Check if batch size is met
    if incoming_request_batches[model_alias].qsize() >= max(allowed_batch_sizes):
        print(f"Moving batch for {model_alias} from incoming to running due to batch size")
        running_request_batches[model_alias] = Queue()
        while not incoming_request_batches[model_alias].empty():
            running_request_batches[model_alias].put(incoming_request_batches[model_alias].get())
        # Process the batch because the batch size was met
        completed_inference_ids = process_batch(model_alias, "Batch size", max(allowed_batch_sizes))

        return jsonify({
        'message': f"f'Inferences completed with {model_alias}: {completed_inference_ids}'"
    })

    return jsonify({
        'message': f"Request queued with ID {request_id} for model {model_alias}"
    })


@app.route('/health')
def health():
    return 'OK', 200


if __name__ == '__main__':

    # Start the background thread to process batches based on time limit
    # threading.Thread(target=background_batch_processor, daemon=True).start()

    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
