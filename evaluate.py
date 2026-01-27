"""
Evaluation script to compare base model vs fine-tuned model outputs.

This helps demonstrate the impact of fine-tuning by showing side-by-side
outputs for the same prompts.
"""
import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Test prompts for evaluation (mix of training and novel)
EVAL_PROMPTS = [
    {
        "id": "from_training",
        "instruction": "Generate QA test cases for the following requirement: Password fields must require at least one special character and be 8 chars long.",
        "source": "training",
    },
    {
        "id": "novel_shipping",
        "instruction": "Generate QA test cases for the following requirement: Orders over $50 must qualify for free shipping.",
        "source": "novel",
    },
    {
        "id": "novel_dark_mode",
        "instruction": "Generate QA test cases for the following requirement: Users must be able to toggle between light and dark mode themes.",
        "source": "novel",
    },
]


def get_latest_adapter_path():
    """Find the latest model adapter directory."""
    if not os.path.exists("models"):
        return None
    model_dirs = [
        d for d in os.listdir("models") 
        if os.path.isdir(f"models/{d}") and d.startswith("qa-fine-tuned")
    ]
    if model_dirs:
        model_dirs.sort(key=lambda d: os.path.getmtime(f"models/{d}"), reverse=True)
        return f"models/{model_dirs[0]}"
    return None


def generate_output(model, tokenizer, device, instruction, max_new_tokens=200):
    """Generate output for a given instruction."""
    prompt = f"User: {instruction}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in full_output:
        return full_output.split("Assistant:")[-1].strip()
    return full_output


def has_bdd_structure(output):
    """Check if output has BDD structure."""
    patterns = [r"Scenario:\s*\w+", r"(Given|When|Then)\s+"]
    return any(re.search(p, output, re.IGNORECASE) for p in patterns)


def main():
    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    print("=" * 70)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    base_model.to(device)
    base_model.eval()
    
    # Check for fine-tuned model
    adapter_path = get_latest_adapter_path()
    fine_tuned_model = None
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading fine-tuned adapter from: {adapter_path}")
        # Need to reload base for PEFT
        base_for_peft = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )
        fine_tuned_model = PeftModel.from_pretrained(base_for_peft, adapter_path)
        fine_tuned_model.to(device)
        fine_tuned_model.eval()
    else:
        print("No fine-tuned model found. Run train.py first.")
        print("Showing base model outputs only.\n")
    
    # Evaluate
    results = []
    
    for prompt in EVAL_PROMPTS:
        print("=" * 70)
        print(f"Prompt ID: {prompt['id']} (source: {prompt['source']})")
        print(f"Instruction: {prompt['instruction'][:80]}...")
        print("-" * 70)
        
        # Base model output
        base_output = generate_output(base_model, tokenizer, device, prompt["instruction"])
        base_has_bdd = has_bdd_structure(base_output)
        
        print("\n[BASE MODEL OUTPUT]")
        print(base_output[:500])
        print(f"\nHas BDD structure: {base_has_bdd}")
        
        result = {
            "id": prompt["id"],
            "source": prompt["source"],
            "instruction": prompt["instruction"],
            "base_output": base_output,
            "base_has_bdd": base_has_bdd,
        }
        
        # Fine-tuned model output
        if fine_tuned_model:
            ft_output = generate_output(fine_tuned_model, tokenizer, device, prompt["instruction"])
            ft_has_bdd = has_bdd_structure(ft_output)
            
            print("\n[FINE-TUNED MODEL OUTPUT]")
            print(ft_output[:500])
            print(f"\nHas BDD structure: {ft_has_bdd}")
            
            result["fine_tuned_output"] = ft_output
            result["fine_tuned_has_bdd"] = ft_has_bdd
            result["improvement"] = ft_has_bdd and not base_has_bdd
        
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    if fine_tuned_model:
        base_bdd_count = sum(1 for r in results if r["base_has_bdd"])
        ft_bdd_count = sum(1 for r in results if r.get("fine_tuned_has_bdd", False))
        improvements = sum(1 for r in results if r.get("improvement", False))
        
        print(f"Base model BDD outputs: {base_bdd_count}/{len(results)}")
        print(f"Fine-tuned BDD outputs: {ft_bdd_count}/{len(results)}")
        print(f"Improvements (gained BDD): {improvements}")
        
        # Save results
        os.makedirs("logs", exist_ok=True)
        eval_file = "logs/evaluation_results.json"
        with open(eval_file, "w") as f:
            json.dump({
                "adapter_path": adapter_path,
                "device": device,
                "results": results,
                "summary": {
                    "base_bdd_count": base_bdd_count,
                    "fine_tuned_bdd_count": ft_bdd_count,
                    "total_prompts": len(results),
                }
            }, f, indent=2)
        print(f"\nDetailed results saved to: {eval_file}")
    else:
        print("Fine-tuned model not available for comparison.")


if __name__ == "__main__":
    main()
