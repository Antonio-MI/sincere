# SINCERE: Scheduling Inference batches in Confidential Environments for Relaxed Executions

## Problem Stament

### Initial Problem Statement

Design an end to end strategy to schedule inference workloads with confidential virtual machines in such way that confidentiality is maintained between end user, cloud provider and model developer.

End user: sent a request with a prompt and the model in which it should be processed. Doesn’t want to share information about sent data to the model developer.

Cloud provider: receives the request from the end user. Owns the scheduler and the confidential VMs. Allows model developers to deploy their models inside a confidential environment.

Model developer: Owns the model and doesn’t want to share information about the model neither to the client or to the cloud provider.

**Challenges**

Maintain confidentiality inside the cloud server: since the scheduler is own by the cloud provider we want to avoid leakage of information from the VMs to the scheduler, so the cloud provider is not able to monitor anything regarding the models or the data.

Scheduling strategies can vary depending on the attribute to maximize (throughput, latency, etc).

Handle swapping or removing models if requested.

Optimize batching while meeting deadline requirements.


After thinking about proposing a novel problem exclusive to confidential computing and not finding one specific task to tackle, we decide to change the objective.

### Final Problem Statement

Evaluate the performance of latency, throughput, service level agreements (SLAs) attainment and gpu utilization in a inference server within a confidential and non-confiential environment with NVIDIA H100 GPU under various traffic patterns and scheduling strategies. 

We expect to be able to provide an insight on how confidential and non-confidential scenarios differ when it comes to perform inference with Large Language Models (LLMs). The constraint that surround the problem are that we can only use one Virtual Machine (VM), access one GPU, and load one model at a time, therefore the need to control model loading and unloading.

The experiments will simulate different real-world scenarios by varying parameters such as traffic load, traffic distribution patterns, scheduling modes and SLAs.

The parameters that will vary over the experiments are:

- **Input Traffic Pattern**: distribution follow by incoming traffic. Traffic distributions studied are: gamma distribution, bursty, and ramp up/down

- **Traffic Pattern Mean**: mean requests per second within each traffic pattern. To compare results, we want to deal with different distributions that over time have the same average request arrival.

- **Scheduling Strategies**: the strategies according to which the requests will be batched and processed. The strategies proposed are the following:
    - "BestBatch": consist of waiting to fill the batch that yields the maximum throughput for each model, value known after performing the batch profiling explained below. The goal is to set a baseline, aiming only for optimal batch sizes for throughput while ignoring latency and model loading times.
    - "BestBatch+Timer": consits of waiting to fill the batch that yields the maximum throughput for each model and adjusting the time a batch has to wait before being moved for processing to account for model swapping time so if the scheduler sees that the latency constraint is going to be violated procedes to process the batch with its current size, without waiting for the full batch. The goal is met SLAs while mainting a high throughput.
    - "SelectBatch+Timer": consist of out of a set of batch sizes, select the most appropiade one according to part arrivals. For that we know `batch_accumulation_time = batch_size / arrival_rate` and then to met SLA `batch_accumulation_time <= desired_latency` therefore `batch_size <= arrival_rate * desired_latency`. It also adjusts the time a batch has to wait before being moved for processing to account for model swapping time. The goal is optimize to meet SLA better but sacrificing throughput.
    - "BestBatch+PartialBatch+Timer": consists in the same as "BestBatch+Timer" but also processes batches that are not full for current loaded model before swapping. The goal is to minimize swap frequency while trying to met SLAs

- **SLAs**: Time that can pass before considering the request as not achieved by the inference server. The values to explore are 40 and 60 seconds.


## Setup environment

To setup the environment create a new virtual environment where the git repository is located.

```
git clone https://github.com/yourusername/yourrepository.git
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

Model profiling consists of recording model loading and unloading times, along with their sizes and standard deviations computed over several iterations. `profiling_models.py` is used for that purposes, and it saves the results in the folder `profiling_results` in a csv that starts with `"model_loading_times"`. Those results will be used later for scheduling.

Batch profiling consists of performing inference using each model with increasing batch sizes until there is an out of memory error, therefore when the GPU can no longer handle a batch size. During that process a csv containing the columns of model, batch size, processing time, throughput (during inference) and several parameters monitored about cpu and gpu functioning with `monitor.py`. In order to do that we have two scripts: 
(i) `profiling_batch_calls.py` that sends the batches of requests. The batch sizes to try are powers of 2 and several prime numbers.
(ii) `profiling_batch_flask.py` that creates a flask api to receive and process the batches. 

Batch profiling is controlled by `run_profiling.sh`. Within this both scripts are synchronized and the process runs automatically. The user must set a runtime long enough to profile all the batches (up to out of memory) for all models.


## Run experiments

In a similar way to the prior profiling, the experiments are fully controlled by `run_experiments.py`.

The scripts controlled by that are:

(i) `api_calls.py`: A script that simulates incoming requests to the server, following specified traffic patterns and rates.

(ii) `api_scheduler_experiments.py`: A Flask API that handles incoming inference requests, batches them according to specified strategies, and processes them using machine learning models.

## Process results



