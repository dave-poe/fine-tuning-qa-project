import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "models/qa-fine-tuned"

import os
if not os.path.exists(adapter_path):
    print(f"Error: Adapter not found at {adapter_path}.")
    print("Please run 'python train.py' first to generate the model weights.")
    exit(1)

# Auto-detect device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load on CPU first to avoid 'meta' device KeyError on Mac with PEFT
print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading Fine-Tuned Adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print(f"Moving model to {device}...")
model.to(device)

# Test Input
instruction = "Generate QA test cases for the following requirement: The system must prevent users from checking out if the inventory count is less than the requested quantity."
prompt = f"User: {instruction}\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("\nGenerating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=True, 
        temperature=0.7
    )

print("\n### Generated Test Cases ###\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
