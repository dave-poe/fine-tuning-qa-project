import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. Config
model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
max_steps = int(os.environ.get("MAX_STEPS", 100))
output_dir = f"models/qa-fine-tuned-steps-{max_steps}"

# Auto-detect device
if torch.cuda.is_available():
    device = "cuda"
    use_mps = False
elif torch.backends.mps.is_available():
    device = "mps"
    use_mps = True
else:
    device = "cpu"
    use_mps = False

print(f"Using device: {device}")
print(f"Loading model: {model_name}...")

# 2. Data
dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})

# 3. Model & Tokenizer
# Load in float32 for maximum compatibility on Mac MPS
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

# 4. LoRA
peft_config = LoraConfig(
    r=8, # Low rank for speed
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# 5. Pre-process dataset to add 'text' column
# We explicitly add the EOS token so the model learns to STOP talking.
def format_row(row):
    return {"text": f"User: {row['instruction']}\nAssistant: {row['output']}{tokenizer.eos_token}"}

print("Formatting dataset...")
dataset = dataset.map(format_row)

# 6. Training Args (SFTConfig)
training_args = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=15,
    max_steps=max_steps,
    use_mps_device=use_mps,
    fp16=False,
    bf16=False,
    report_to="none",
    packing=False
)

# 7. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    args=training_args,
)

print(f"Starting training on {device}...")
trainer.train()
print("Training finished!")
trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
print(f"\nTraining Summary:")
print(f"  Device: {device}")
print(f"  Base Model: {model_name}")
print(f"  Steps Completed: {trainer.state.global_step}")
print(f"  Model Location: {output_dir}")
