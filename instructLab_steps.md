### Steps to generate prompts with InstructLab 

https://github.com/instructlab/instructlab?tab=readme-ov-file

https://github.com/instructlab/taxonomy

Using a Mac:

- Create a folder for instruct lab and install

  ```bash
  mkdir instructlab
  cd instructlab
  python3 -m venv --upgrade-deps venv
  source venv/bin/activate
  pip cache remove llama_cpp_python
  pip install 'instructlab[mps]'
  ```

- Verify install with ``ilab`` and initialize with ``ilab config init``. The continue the steps given

- Run ``ilab model download``

- To serve the model and chat with it ``ilab model serve`` on one terminal and ``ilab model chat``

- To activate the environment needed for this remember: from the instructlab directory: ``source venv/bin/activate`` 

- In order to create prompts we have to generate a new skill. For this purpose we will go in instruct lab > taxonomy > compositional_skills > writing > freeform and there we create a new folder.

- Inside that folder we create a yaml file called qna.yaml. That file follows this structure:

- ```yaml
  version: 2
  task_description: 'Example prompts to generate more prompts'
  created_by: IBM Research
  seed_examples:
  - answer: The closest planet to the Sun is Mercury
    question: Which plantet is the closest to the Sun?
  
  - answer: The sky is blue because blue light is scattered more than other colors due to its shorter wavelength
    question: Why is the sky blue?
  
  - answer: The Amazon Rainforest is located in South America
    question: Where is the Amazon Rainforest located?
  ```

- Running ``ilab data generate`` starts the process to create prompts. That will generate three files in instruct lab > generated.

- One file is a json with the format (e.g. generated_merlinite-7b-lab-Q4_K_M_2024-08-13T14_24_19.json):

  ```json
  [
      {
          "instruction": "Why do some people enjoy cooking so much?",
          "input": "Cooking can be a fulfilling and creative way to express oneself. It allows you to combine various ingredients and flavors to create something new, which can lead to a sense of accomplishment and satisfaction.",
          "output": "Some people find joy in the act of creating delicious meals that nourish both body and soul. The process of cooking can be a form of self-expression and personal growth, as one learns how different ingredients interact with each other and how various techniques affect the final dish. This sense of accomplishment and creativity can make cooking an enjoyable and fulfilling pastime for many people.",
          "taxonomy_path": "compositional_skills->writing->freeform->prompts",
          "task_description": "Example prompts to generate more prompts",
          "document": null
      },
      {
          "instruction": "How can you tell if a person has autism?",
          "input": "",
          "output": "Autism is a spectrum disorder and can manifest differently in each individual. However, some common signs include difficulty with social interactions, repetitive behaviors, and sensory sensitivities. It is essential to remember that only a professional evaluation can determine if someone has autism.",
          "taxonomy_path": "compositional_skills->writing->freeform->prompts",
          "task_description": "Example prompts to generate more prompts",
          "document": null
      }
  ]
  ```

- And another two called train and test with extension .jsonl (e.g. train_merlinite-7b-lab-Q4_K_M_2024-08-13T14_24_19.jsonl):

  ```json
  
  {"system": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.", "user": "Which plantet is the closest to the Sun?", "assistant": "The closest planet to the Sun is Mercury"}
  {"system": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.", "user": "Why is the sky blue?", "assistant": "The sky is blue because blue light is scattered more than other colors due to its shorter wavelength"}
  {"system": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.", "user": "Where is the Amazon Rainforest located?", "assistant": "The Amazon Rainforest is located in South America"}
  {"system": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.", "user": "What is the world's largest coral reef system?", "assistant": "The Great Barrier Reef is the world's largest coral reef system"}
  {"system": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.", "user": "What causes lightning?", "assistant": "Lightning is a natural electrical discharge caused by imbalances between storm clouds and the ground"}
  {"system": "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.", "user": "What is the largest hot desert in the world?", "assistant": "The Sahara Desert is the largest hot desert in the world"}
  ```
  
  



