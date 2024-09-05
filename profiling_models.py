
import time
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import platform

# Folder containing models
base_dir = "./models"

# Select device, cpu for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Save machine name to identify csv
machine_name = platform.node()

# Dictionaries to store mean load and unload times
model_load_times = {}
model_unload_times = {}

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

# Directory for models
os.makedirs("./outputs", exist_ok=True)
results = []

# Iterate through all the models available to profile their mean load and unload
for model in ["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m"]:
    load_times = []
    unload_times = []
    model_sizes = []

    for _ in range(10):
        # Profile the loading time
        load_start_time = time.time()
        model_dir = os.path.join(base_dir, model)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model_instance = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        load_time = time.time() - load_start_time
        load_times.append(load_time)

        # Calculate model size (parameters + tokenizer)
        model_param_size = get_model_size_in_bytes(model_instance)
        tokenizer_size = get_tokenizer_size_in_bytes(model_dir)
        total_model_size = model_param_size + tokenizer_size
        model_sizes.append(total_model_size)

        # Profile the unloading time
        unload_start_time = time.time()
        del model_instance
        del tokenizer
        if device=="cuda": 
            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
        unload_time = time.time() - unload_start_time
        unload_times.append(unload_time)

    # Calculate the mean load and unload times
    mean_load_time = round(np.mean(load_times), 4)
    mean_unload_time = round(np.mean(unload_times), 4)
    model_size = round(np.mean(model_sizes) / (1024**3), 2)  # Convert bytes to GB

    std_load_time = round(np.std(load_times), 4)
    std_unload_time = round(np.std(unload_times), 4)

    results.append((
                        model, model_size, mean_load_time, std_load_time, mean_unload_time, std_unload_time
                    ))

    # Store the mean times in the dictionaries
    model_load_times[model] = mean_load_time
    model_unload_times[model] = mean_unload_time

    print(f"Profiled model {model} - Load time: {mean_load_time:.4f}s, Unload time: {mean_unload_time:.4f}s")

# Save model profiling info into a csv
df = pd.DataFrame(
            results,
            columns=[
                "model_name", "model_size /GB", "mean_loading_time /s", "std_loading_time /s", "mean_unloading_time /s", "std_unloading_time /s"
            ],
            )

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
csv_filename = f"model_loading_times_{machine_name}_{device}_{timestamp}.csv"
csv_path = "outputs/" + csv_filename
file_exists = os.path.isfile(csv_path)
if file_exists:
    df.to_csv(csv_path, mode="a", header=False, index=False)
else:
    df.to_csv(csv_path, index=False)