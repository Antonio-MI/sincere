# SINCERE: Scheduling Inference batches in Confidential Environments for Relaxed Executions

### Problem Stament

Schedule relaxed inference workloads while maintaining integrity & confidentiality between:
End User: Sends prompts and specifies models without revealing prompts to the model developer or cloud provider
Cloud Provider: Owns the scheduler and confidential VM platform. It should not know scheduling patterns and frequency of model usage
Model Developer: Wants to maintain model confidentiality from end users and cloud providers.  
