import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

model_name = "Qwen/Qwen3-0.6B-Instruct"
adapter_path = "models/qa-fine-tuned"

# Try to find the latest model directory (with steps in name)
model_dirs = [d for d in os.listdir("models") if os.path.isdir(f"models/{d}") and d.startswith("qa-fine-tuned")]
if model_dirs:
    # Sort by modification time, get the latest
    model_dirs.sort(key=lambda d: os.path.getmtime(f"models/{d}"), reverse=True)
    adapter_path = f"models/{model_dirs[0]}"

# Extract steps from directory name if available
steps_match = re.search(r'steps-(\d+)', adapter_path)
training_steps = steps_match.group(1) if steps_match else "Unknown"

import os
if not os.path.exists(adapter_path):
    print(f"Error: Adapter not found at {adapter_path}.")
    print("Please run 'python train.py' first to generate the model weights.")
    exit(1)

# Load on CPU first to avoid 'meta' device KeyError on Mac with PEFT
print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading Fine-Tuned Adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Now move to MPS (Metal) for fast generation
device = "mps" if torch.backends.mps.is_available() else "cpu"
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

print("\n" + "="*60)
print("Model Information:")
print(f"  Base Model: {model_name}")
print(f"  Training Steps: {training_steps}")
print(f"  Adapter Path: {adapter_path}")
print("="*60)
