from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import time
import numpy as np

# Folder containing models
base_dir = "./models"

# Select device, cpu for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# To save current model loaded name and model, and its tokenizer
current_model_alias = None
loaded_model = None
loaded_tokenizer = None

# Dictionaries to store load and unload times
model_load_times = {}
model_unload_times = {}

#Function to load models
def load_model(model_alias):
    global current_model_alias, loaded_model, loaded_tokenizer

    print(f"{model_alias} load time {model_load_times[model_alias]}")

    # If the model requested is the one already in memory, no need to do something else
    if model_alias == current_model_alias:
        print("Model already loaded")
        return

    # If the model requested is not loaded, delete from memory the one that is loaded
    if loaded_model is not None:
        del loaded_model
        del loaded_tokenizer
        print("Model unloaded")
        #torch.cuda.empty_cache() #if using GPU 

    # Load new model and its tokenizer
    model_name = model_alias
    model_dir = os.path.join(base_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    print("Model loaded")

    # Uptade parameters
    current_model_alias = model_alias
    loaded_model = model
    loaded_tokenizer = tokenizer

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    # Read model requested and prompt from the json sent by the EndUser
    model_alias = request.json.get('model_alias')
    prompt = request.json.get('prompt')

    # Call to load the model
    load_model(model_alias)

    # Pipeline to perform inference
    pipe = pipeline(
        "text-generation",
        model=loaded_model,
        tokenizer=loaded_tokenizer,
        device="cpu",  # replace with "mps" to run on a Mac device
        torch_dtype = torch.float16
    )

    # Call for inference
    outputs = pipe(prompt, max_new_tokens=128)
    #response = outputs[0]["generated_text"]

    #return jsonify({'response': response})
    return jsonify({'response': "Inference completed"})

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
            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
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

