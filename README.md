# QA Fine-Tuning Project

A hands-on project demonstrating how to fine-tune a Large Language Model (LLM) to specialize in converting **Product Requirements** into **Structured Test Cases** (Given-When-Then format).

This project showcases the end-to-end MLOps pipeline:
1.  **Data Engineering**: Synthetically generating domain-specific training data.
2.  **Fine-Tuning**: Using LoRA (Low-Rank Adaptation) and PEFT to train efficiently on consumer hardware.
3.  **Inference**: Running the specialized model to generate valid test cases.

## Project Structure

- `data/`: Contains the training (`train.jsonl`) and validation (`val.jsonl`) datasets.
- `models/`: Directory where fine-tuned weights are saved (ignored in git).
- `train.py`: Main script to fine-tune the model using HuggingFace TRL/PEFT.
- `inference.py`: Script to load the base model + adapter and run inference.
- `populate_data.py`: Script used to generate synthetic training data.
- `requirements.txt`: Python dependencies.

## Installation

### Prerequisites
- Python 3.10+
- A machine with a GPU (NVIDIA or Mac M-series with MPS support is recommended).

### Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/fine_tuning_qa_project.git
    cd fine_tuning_qa_project
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Generation (Optional)
The repository includes sample data in `data/`. If you want to regenerate it or create more:
```bash
python populate_data.py
```
*Note: This may require an OpenAI API key or similar if using an external "Teacher Model".*

### 2. Fine-Tuning
Start the training process. This uses Qwen2.5-0.5B-Instruct as the base model and applies LoRA adapters.
```bash
python train.py
```
This will run for the configured number of epochs/steps and save the adapter weights to `models/qa-fine-tuned`.

### 3. Inference
Test the trained model with a new requirement:
```bash
python inference.py
```
Modify the `instruction` variable in `inference.py` to test different inputs.

## Technical Details

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct` (Small, efficient, good instruction following capability).
- **Technique**: LoRA (Low-Rank Adaptation) via `peft`.
- **Frameworks**: `transformers`, `trl` (Transformer Reinforcement Learning library for SFT), `accelerate`.
- **Hardware Optimization**: Configured to run on Mac (MPS) or CUDA if available.

## License
MIT
