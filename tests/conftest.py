"""
Pytest fixtures for model validation tests.
"""
import os
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "models/qa-fine-tuned"


@pytest.fixture(scope="session")
def device():
    """Detect and return the appropriate device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="session")
def tokenizer():
    """Load and return the tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="session")
def model(device):
    """Load the fine-tuned model with LoRA adapter."""
    if not os.path.exists(ADAPTER_PATH):
        pytest.skip(f"Model not found at {ADAPTER_PATH}. Run training first.")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    
    # Load fine-tuned adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()
    
    return model


@pytest.fixture(scope="session")
def generate_output(model, tokenizer, device):
    """Factory fixture that returns a function to generate model output."""
    
    def _generate(instruction: str, max_new_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate output for a given instruction."""
        prompt = f"User: {instruction}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "Assistant:" in full_output:
            return full_output.split("Assistant:")[-1].strip()
        return full_output
    
    return _generate
