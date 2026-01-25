# QA Fine-Tuning Project

This project demonstrates fine-tuning using Low-Rank Adaptation (LoRA) with Qwen3-0.6B-Instruct to specialize in converting **Product Requirements** into **Structured BDD Test Cases** (Given-When-Then format).

Showcases the end-to-end MLOps pipeline with automated testing and GitHub Actions workflows:
1. **Data Engineering**: Domain-specific training data in BDD format (60 examples).
2. **Fine-Tuning**: Efficient training using LoRA/PEFT on consumer hardware.
3. **Validation**: Comprehensive pytest suite with HTML/JUnit reporting.
4. **Inference**: Run specialized model with automatic model versioning.
5. **CI/CD**: GitHub Actions for automated training and validation.

## Project Structure

- `data/`: Training (`train.jsonl`) and validation (`val.jsonl`) datasets - all BDD format
- `models/`: Fine-tuned model directories, named by training steps
- `tests/`: Pytest validation suite with 37 tests covering structure, keywords, quality, and consistency
- `train.py`: Fine-tuning script with device auto-detection and training summary
- `inference.py`: Inference script with model metadata output
- `.github/workflows/`: GitHub Actions for training and inference with test reporting

## Installation

### Prerequisites
- Python 3.11+
- GPU recommended (NVIDIA CUDA or Mac M-series with MPS)

### Setup
```bash
# Clone and setup
git clone https://github.com/dave-poe/fine-tuning-qa-project.git
cd fine-tuning-qa-project

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```
- Runs for configured steps (default: 50)
- Saves model as `models/qa-fine-tuned-steps-{steps}`
- Shows training summary with device, steps completed, and model location

### Inference
```bash
python inference.py
```
- Auto-detects latest model
- Displays model metadata (base model, training steps, adapter path)
- Shows generated BDD test cases

### Running Tests
```bash
pytest tests/test_model_validation.py -v
```
- 37 tests covering structure, keywords, quality, consistency
- Generates JUnit XML and HTML reports
- 80% consistency threshold (5 runs)

See `tests/README.md` for comprehensive testing documentation.

## Model Information

| Property | Value |
|----------|-------|
| **Base Model** | Qwen/Qwen3-0.6B-Instruct |
| **Architecture** | 600M parameters |
| **Training Method** | LoRA (Low-Rank Adaptation) |
| **Training Data** | 50 BDD examples |
| **Validation Data** | 10 BDD examples |
| **Output Format** | Given/When/Then test cases |

## Key Features

✅ Consistent BDD-formatted training data (all 60 examples)
✅ Comprehensive validation suite (37 tests)
✅ Automatic model versioning by training steps
✅ GitHub Actions workflows with test reporting
✅ Device auto-detection (CUDA/MPS/CPU)
✅ 80% consistency requirement for robustness

## CI/CD Workflows

### Fine-Tune Workflow
- Trigger: Manual or scheduled
- Configurable training steps (default: 50)
- Uploads model as artifact (30-day retention)

### Inference Workflow
- Auto-triggers after training completes
- Manual trigger with optional run ID
- Runs validation suite with HTML and JUnit reporting
- GitHub Test Reporter integration

## License

MIT
