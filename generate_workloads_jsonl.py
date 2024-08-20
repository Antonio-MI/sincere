import json
import os
import random

# Path to the directory containing your input JSONL files
input_directory = 'instructlab/generated'

# Define the model aliases
# model_aliases = ["gpt2-124m", "distilgpt2-124m", "gptneo-125m", "gpt2medium-355m"]
model_aliases = ["gpt2medium-355m"]

# Create a directory to store the generated JSON files
output_directory = './workloads'
os.makedirs(output_directory, exist_ok=True)

def process_jsonl_file(file_path, output_directory, start_index=1):
    """Processes a single JSONL file and creates output JSON files."""
    with open(file_path, 'r') as file:
        for line_index, line in enumerate(file):
            try:
                # Parse each line as a JSON object
                data = json.loads(line)
                
                # Extract the "user" field and rename it to "prompt"
                if "user" in data:
                    prompt = data["user"]
                    
                    # Create the new JSON structure
                    new_json = {
                        "prompt": prompt,
                        "model_alias": random.choice(model_aliases)
                    }
                    
                    # Define the output file name
                    output_file_name = f'workload{start_index + line_index}.json'
                    output_file_path = os.path.join(output_directory, output_file_name)
                    
                    # Write the new JSON to a file
                    with open(output_file_path, 'w') as output_file:
                        json.dump(new_json, output_file, indent=4)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON in file: {file_path}, line: {line_index + 1}")

def process_all_files_in_directory(input_directory, output_directory):
    """Processes all JSONL files in a directory."""
    jsonl_files = [f for f in os.listdir(input_directory) if f.endswith('.jsonl')]
    total_index = 1
    
    for jsonl_file in jsonl_files:
        file_path = os.path.join(input_directory, jsonl_file)
        print(f"Processing file: {file_path}")
        process_jsonl_file(file_path, output_directory, start_index=total_index)
        
        # Update the starting index for the next file to avoid name collisions
        total_index += sum(1 for line in open(file_path, 'r'))

if __name__ == "__main__":
    process_all_files_in_directory(input_directory, output_directory)
    print(f"Generated JSON files are saved in {output_directory}")
