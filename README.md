# SINCERE: Scheduling Inference batches in Confidential Environments for Relaxed Executions

### Problem Stament

Schedule relaxed inference workloads while maintaining integrity & confidentiality between:
End User: Sends prompts and specifies models without revealing prompts to the model developer or cloud provider
Cloud Provider: Owns the scheduler and confidential VM platform. It should not know scheduling patterns and frequency of model usage
Model Developer: Wants to maintain model confidentiality from end users and cloud providers.  


### Code

#### download_models.py

Downloads the set of models specified from Huggingface inside `models`. It requieres a HuggingfaceToken (read permission) and in some cases a previous request for access to the repository containing the model itself.

#### profiling_models.py

For each model in the list defined at the beginning, loads and unloads the model 10 times, and saves a csv containing the model name, its size in GB, the mean loading time in seconds and its standard deviation, the mean unloading time and its standard deviation.

#### generate_workloads_jsonl.py

Takes a folder that contains workloads generated with instructlab (jsonl) and outputs a json for each one of the list contained in the jsonl with the following sctructure. The script randomly assigns a model out of a list to the prompt. That model can later be overwritten to have more control over the calls.
```
{
    "prompt": "What is an example of a popular tourist destination in Australia that offers unique experiences for visitors?",
    "model_alias": "gpt2-124m"
}
```

#### profiling_batch_calls.py + profiling_batch_flask.py
...

#### api_calls_async + api_scheduler.py
...


