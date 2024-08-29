import sys
print(sys.executable)

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

def download_and_save_model(model_name, model_dir):
    # Directory for models
    os.makedirs(model_dir, exist_ok=True)

    # Download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save model and tokenizer
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)



if __name__ == "__main__":

    login(token='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')

    models = {
        "gpt2-124m" : "openai-community/gpt2",
        "distilgpt2-124m": "distilbert/distilgpt2",
        "gptneo-125m": "EleutherAI/gpt-neo-125m",
        "gpt2medium-355m": "openai-community/gpt2-medium"
        #"granite-7b": "ibm-granite/granite-7b-base", 
        #"gemma-7b": "google/gemma-7b",
        #"llama3-8b": "meta-llama/Meta-Llama-3.1-8B", # Not enough size
        #"llama3-70b": "meta-llama/Meta-Llama-3-8B" # Too big
    }

    base_dir = "./models"

    for model_alias, model_name in models.items():
        model_dir = os.path.join(base_dir, model_alias)
        download_and_save_model(model_name, model_dir)
