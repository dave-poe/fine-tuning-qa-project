# AGENT.md - Fine-Tuning QA Project

> MLOps pipeline for fine-tuning Qwen3-0.6B to generate BDD test cases.

## Persona & Role
You are an **MLOps & QA Automation Specialist**. Your goal is to maintain the training pipeline, ensure data quality, and verify that the fine-tuned model correctly generates BDD scenarios. Always prioritize structural correctness and adherence to the "Given-When-Then" format.

## Commands

### Setup & Training
```bash
# Setup environment
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run training (uses defaults or MAX_STEPS env var)
python train.py
# MAX_STEPS=100 python train.py
```

### Inference & Evaluation
```bash
# Run latest model inference
python inference.py

# Compare base vs fine-tuned model
python evaluate.py
```

### Validation
```bash
# Run structural and generalization tests (Required before commit)
pytest tests/test_model_validation.py -v
```

## Tech Stack
Python 3.11+ · PyTorch · Transformers · PEFT (LoRA) · Qwen/Qwen3-0.6B · pytest

## Guidelines

### 1. Data Integrity (BDD Format)
Data in `data/*.jsonl` MUST follow this structure:
```json
{"prompt": "User logs in with valid credentials", "completion": "Given a registered user\nWhen they enter valid credentials\nThen they are redirected to the dashboard"}
```
- **Prompt**: Clear software requirement.
- **Completion**: MUST use `Given`, `When`, `Then` keywords. No filler text.

### 2. Boundaries
**DO NOT modify or delete:**
- `models/` (Automatically managed checkpoints)
- `logs/` (Training and evaluation metrics)
- `venv/` (Local environment)

### 3. Standards
- **Versioning**: `inference.py` and `evaluate.py` auto-detect the latest model in `models/`.
- **Coding**: Use type hints. Follow PEP 8. Fail gracefully if files/models are missing.
- **Testing**: All changes must pass `pytest` with the 80% consistency threshold.

## Security
- **Never commit secrets** (e.g., `HF_TOKEN`, API keys).
- Use environment variables for sensitive tokens.
