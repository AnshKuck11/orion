from transformers import AutoTokenizer, AutoModelForCausalLM #imports the necessary libraries from the transformers package to load the tokenizer and model for natural language processing tasks.
import torch # imports the PyTorch library, which is used for tensor computations and deep learning tasks.

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # sets the name of the pre-trained model to be used, which is "TinyLlama/TinyLlama-1.1B-Chat-v1.0". This model is a smaller version of the LLaMA model, designed for chat applications.

print("Loading Orion's Brain") 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # loads the tokenizer associated with the specified model. The tokenizer is responsible for converting text into a format that the model can understand (tokenization) and vice versa (detokenization). It ensures that the input text is properly formatted for the model's architecture.
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="cuda") # loads the pre-trained model for causal language modeling (used for generating text) from the specified model name. The model is loaded with a data type of float16 to optimize memory usage and is mapped to run on a CUDA-enabled GPU for faster inference. This allows the model to generate responses based on the input it receives.

print("Orion's Brain loaded successfully")

messages = [ 
    {"role": "system", "content": "You are Orion, a helpful personal AI assistant."},
    {"role": "user", "content": "Hey Orion, introduce yourself!"}
] # defines a list of messages that will be used as input for the model. The messages are structured in a way that indicates the role of each message (system or user) and the content of the message. In this case, the system message sets the context for the model, while the user message is a prompt asking Orion to introduce itself.

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature = 0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Orion: " + response)

