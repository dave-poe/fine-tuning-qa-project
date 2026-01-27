# QA Fine-Tuning Project

This project demonstrates fine-tuning using Low-Rank Adaptation (LoRA) with Qwen3-0.6B to specialize in converting **Product Requirements** into **Structured BDD Test Cases** (Given-When-Then format).

Showcases the end-to-end MLOps pipeline with automated testing and GitHub Actions workflows:

1. **Data Engineering**: Domain-specific training data in BDD format (40 training + 10 validation + 8 test examples).
2. **Fine-Tuning**: Efficient training using LoRA/PEFT on consumer hardware.
3. **Validation**: Comprehensive pytest suite with HTML/JUnit reporting.
4. **Inference**: Run specialized model with automatic model versioning.
5. **Evaluation**: Compare base model vs fine-tuned model outputs.
6. **CI/CD**: GitHub Actions for automated training and validation.

## Project Structure

- `data/`: Training, validation, and test datasets - all BDD format
  - `train.jsonl`: 40 unique training examples
  - `val.jsonl`: 10 validation examples (different from training)
  - `test.jsonl`: 8 held-out test examples for final evaluation
- `models/`: Fine-tuned model directories, named by training steps
- `logs/`: Training metrics and evaluation results
- `tests/`: Pytest validation suite covering structure, keywords, quality, and generalization
- `train.py`: Fine-tuning script with device auto-detection and metrics logging
- `inference.py`: Inference script with model metadata output
- `evaluate.py`: Compare base model vs fine-tuned model outputs
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

- Runs for configured steps (default: 100, configurable via `MAX_STEPS` env var)
- Saves model as `models/qa-fine-tuned-steps-{steps}`
- Logs training metrics to `logs/training_metrics_{steps}_steps.json`
- Shows training summary with device, steps completed, and model location

### Inference

```bash
python inference.py
```

- Auto-detects latest model
- Displays model metadata (base model, training steps, adapter path)
- Shows generated BDD test cases

### Evaluation

```bash
python evaluate.py
```

- Compares base model vs fine-tuned model outputs
- Tests on both training-based and novel prompts
- Saves detailed results to `logs/evaluation_results.json`

### Running Tests

```bash
pytest tests/test_model_validation.py -v
```

- Tests covering structure, keywords, quality, consistency
- Includes generalization tests with novel prompts not seen in training
- Generates JUnit XML and HTML reports
- 80% consistency threshold (5 runs)

See `tests/README.md` for comprehensive testing documentation.

## Model Information

| Property | Value |
|----------|-------|
| **Base Model** | Qwen/Qwen3-0.6B |
| **Architecture** | 600M parameters |
| **Training Method** | LoRA (Low-Rank Adaptation) |
| **Training Data** | 40 unique BDD examples |
| **Validation Data** | 10 unique BDD examples |
| **Test Data** | 8 held-out BDD examples |
| **Output Format** | Given/When/Then test cases |

## Training Data Domains

The training data covers diverse software testing scenarios:

- User authentication (login, logout, SSO, MFA)
- Form validation (email, phone, dates, file uploads)
- API behaviors (rate limiting, pagination, error handling)
- E-commerce (cart, checkout, payments, inventory)
- Data operations (CRUD, search, filtering, sorting)
- Security (authorization, encryption, SQL injection)
- Performance (timeouts, lazy loading, caching)
- User preferences (notifications, language, themes)

## Key Features

- Diverse BDD-formatted training data (40 unique examples across multiple domains)
- Comprehensive validation suite with generalization tests
- Training metrics logging for loss curve analysis
- Base vs fine-tuned model comparison script
- Automatic model versioning by training steps
- GitHub Actions workflows with test reporting
- Device auto-detection (CUDA/MPS/CPU)
- 80% consistency requirement for robustness

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
- Requires `HF_TOKEN` secret for model downloads

## Limitations

This is a learning/demonstration project with the following limitations:

- **Dataset Size**: 40 training examples is small for production fine-tuning. Real-world projects typically use hundreds or thousands of examples.
- **Domain Coverage**: While diverse, the training data doesn't cover all possible requirement types.
- **Evaluation**: No formal metrics like BLEU/ROUGE scores. Evaluation is primarily structural.
- **Single Model**: Only tested with Qwen3-0.6B. Results may vary with other base models.

## License

MIT
