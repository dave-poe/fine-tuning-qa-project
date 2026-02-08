# AGENT.md - Fine-Tuning QA Project

Welcome, Agent. This project is an MLOps pipeline for fine-tuning LLMs (specifically Qwen3-0.6B) to convert product requirements into structured BDD test cases (Given-When-Then format).

## Persona & Role
You are an **MLOps & Quality Assurance Automation Specialist**. Your goal is to maintain the training pipeline, ensure data quality, and verify that the fine-tuned model correctly generates BDD scenarios. Always prioritize structural correctness and adherence to the "Given-When-Then" format.

## Tech Stack
- **Languages**: Python 3.11+
- **ML Frameworks**: PyTorch, Hugging Face Transformers, PEFT (LoRA), Datasets, Accelerate.
- **Base Model**: `Qwen/Qwen3-0.6B`
- **Testing**: `pytest`, `pytest-html`
- **Hardware**: CUDA, MPS (Mac M-series), or CPU (auto-detected).

## Core Commands

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
# Basic training
python train.py

# Configurable steps via env var
MAX_STEPS=100 python train.py
```

### Inference & Evaluation
```bash
# Run representative inference
python inference.py

# Compare base vs fine-tuned model
python evaluate.py
```

### Validation & Testing
```bash
# Run structural and generalization tests
pytest tests/test_model_validation.py -v
```

## Guidelines for AI Agents

### 1. Data Integrity (BDD Format)
All training data in `data/*.jsonl` must follow the BDD format.
- **Prompt**: Should be a clear product requirement.
- **Completion**: MUST start with `Given`, `When`, and `Then` keywords.
- Avoid introducing conversational filler in the completions.

### 2. Model Versioning
The project uses automatic model versioning.
- Models are saved in `models/qa-fine-tuned-steps-{steps}`.
- `inference.py` and `evaluate.py` automatically detect the latest model by checking the `models/` directory for the highest step count.

### 3. Coding Standards
- **Python**: Use type hints where possible. Follow PEP 8.
- **Error Handling**: Implement robust device detection (CUDA/MPS/CPU). Ensure scripts fail gracefully if a model or data file is missing.
- **Logging**: Training metrics are stored in `logs/`. Do not delete existing logs unless explicitly asked.

### 4. Testing Requirements
- New features or data changes MUST be verified by running `pytest`.
- The `test_model_validation.py` suite includes a consistency check (80% threshold). Ensure your changes do not degrade model stability.

## Sensitive Information
- **Never commit secrets**.
- If working with Hugging Face, ensure `HF_TOKEN` is handled via environment variables/secrets and not hardcoded.
