# SINCERE: Scheduling Inference batches in Confidential Environments for Relaxed Executions

## Problem Stament

#### Initial Problem Statement
Schedule relaxed inference workloads while maintaining integrity & confidentiality between:
End User: Sends prompts and specifies models without revealing prompts to the model developer or cloud provider
Cloud Provider: Owns the scheduler and confidential VM platform. It should not know scheduling patterns and frequency of model usage
Model Developer: Wants to maintain model confidentiality from end users and cloud providers.  

After thinking about proposing a novel problem and not finding one, we decide to change the objective

#### Final Problem Statement
Using several scheduling strategies, observe how latency, throughput, sla attainment and gpu utilization for inference behave for those strategies for several input traffic patterns within a confidential and non confiential environment with NVIDIA H100


## Setup environment

To setup the environment create a new virtual environment where the git repository is located.

```
cd sincere
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Setup Huggingface models and workloads

The models used in this work have been downloaded from Huggingface. To access the models a HuggingfaceToken (read permission) is requiered and in some cases a previous request for access to the repository containing the model. 

The script to download the models is `download_models.py` and there the token should be defined plus a list of models to download with a model alias and the official model name that can be found at https://huggingface.co/models

In this work "ibm-granite/granite-7b-base", "google/gemma-7b", "meta-llama/Meta-Llama-3.1-8B" were the models used.

The workload used as input for the models has been generated with instructlab (https://github.com/instructlab) as jsonl files. instructlab use is detailed in `instructLab_steps.md`. Once the jsonl has been created, we use `generate_workloads_jsonl.py` which takes a folder that contains workloads generated with instructlab and outputs a json for each list contained in the jsonl with the following sctructure:
```
{
    "prompt": "What is an example of a popular tourist destination in Australia that offers unique experiences for visitors?",
    "model_alias": "gpt2-124m"
}
```
The initial version of the script randomly assigns a model out of a list to the prompt. That model will be overwritten later to have more control over the model distribution.

## Run model and batch profiling

profiling_models.py

For each model in the list defined at the beginning, loads and unloads the model 10 times, and saves a csv containing the model name, its size in GB, the mean loading time in seconds and its standard deviation, the mean unloading time and its standard deviation.

## Run experiments


## Process results



