# Model Validation Test Suite

This test suite validates the fine-tuned QA test case generator model. Tests are designed to handle non-deterministic LLM outputs using pattern matching, keyword relevance, and statistical thresholds rather than exact comparisons.

## Overview

The validation suite ensures the fine-tuned model:

1. Produces properly structured test case output
2. Mentions relevant keywords from requirements
3. Meets quality standards (length, readability, no repetition)
4. Performs consistently across multiple runs

## File Structure

```text
tests/
├── README.md              # This file
├── __init__.py            # Package marker
├── conftest.py            # Pytest fixtures (model, tokenizer, generate_output)
├── test_data.py           # Golden test cases and configuration
└── test_model_validation.py  # Validation test classes
```

## Running Tests

### Prerequisites

- Python 3.11+
- Trained model at `models/qa-fine-tuned/`
- Dependencies: `pip install -r requirements.txt`

### Run All Tests

```bash
pytest tests/test_model_validation.py -v
```

### Run Specific Test Class

```bash
# Structure tests only
pytest tests/test_model_validation.py::TestOutputStructure -v

# Keyword tests only
pytest tests/test_model_validation.py::TestKeywordRelevance -v

# Quality tests only
pytest tests/test_model_validation.py::TestOutputQuality -v

# Consistency tests only
pytest tests/test_model_validation.py::TestConsistency -v
```

### Run Single Test Case

```bash
pytest tests/test_model_validation.py::TestOutputStructure::test_output_contains_test_patterns[password_validation] -v
```

## Test Classes

### TestOutputStructure

Validates that model output contains proper test case structure.

| Test                                  | Description                         | Pass Criteria                                                                                            |
| ------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `test_output_contains_test_patterns`  | Checks for valid test case patterns | Output contains one of: `Test:`, `Scenario:`, `Given/When/Then`, `Input:`, `Expected:`, `Request:`, etc. |
| `test_output_has_multiple_test_cases` | Checks for multiple test cases      | Output contains at least 2 occurrences of `Test:` or `Scenario:`                                         |

### TestKeywordRelevance

Validates that model output mentions relevant terms from the input requirement.

| Test                                        | Description              | Pass Criteria                                      |
| ------------------------------------------- | ------------------------ | -------------------------------------------------- |
| `test_output_mentions_requirement_keywords` | Checks keyword presence  | At least 50% of expected keywords appear in output |

### TestOutputQuality

Validates output quality metrics to catch degenerate outputs.

| Test                                 | Description                | Pass Criteria                                      |
| ------------------------------------ | -------------------------- | -------------------------------------------------- |
| `test_output_minimum_length`         | Checks minimum length      | Output >= 50 characters                            |
| `test_output_maximum_length`         | Checks maximum length      | Output <= 2000 characters (no runaway generation)  |
| `test_no_excessive_repetition`       | Detects repetitive output  | No single line appears more than 3 times           |
| `test_output_not_empty_or_gibberish` | Validates readable content | Output has at least 10 words                       |

### TestConsistency

Validates that model performs consistently across multiple runs (handles non-determinism).

| Test                                      | Description             | Pass Criteria                                |
| ----------------------------------------- | ----------------------- | -------------------------------------------- |
| `test_multiple_runs_pass_structure_check` | Runs generation 5 times | At least 80% of runs produce valid structure |

## Golden Test Cases

The suite uses 5 golden test cases covering different requirement types:

| ID                    | Requirement Type   | Expected Output Style               |
| --------------------- | ------------------ | ----------------------------------- |
| `password_validation` | Password rules     | Test/Input/Expected format          |
| `inventory_checkout`  | Business logic     | BDD Scenario/Given/When/Then format |
| `api_authorization`   | API endpoint       | Test/Request/Response format        |
| `email_validation`    | Input validation   | Test/Input/Expected format          |
| `session_timeout`     | Session management | Mixed format                        |

## Configuration

Thresholds and limits are defined in `test_data.py`:

```python
MIN_OUTPUT_LENGTH = 50          # Minimum characters
MAX_OUTPUT_LENGTH = 2000        # Maximum characters
CONSISTENCY_RUNS = 5            # Number of runs for consistency test
CONSISTENCY_PASS_THRESHOLD = 0.8  # 80% must pass
```

## Fixtures

Defined in `conftest.py`:

| Fixture           | Scope   | Description                          |
| ----------------- | ------- | ------------------------------------ |
| `device`          | session | Auto-detects CUDA/MPS/CPU            |
| `tokenizer`       | session | Loads Qwen tokenizer                 |
| `model`           | session | Loads base model + LoRA adapter      |
| `generate_output` | session | Factory function for text generation |

## Adding New Test Cases

1. Add a new entry to `TEST_CASES` in `test_data.py`:

```python
{
    "id": "unique_id",
    "instruction": "Generate QA test cases for: <your requirement>",
    "expected_keywords": ["keyword1", "keyword2"],
    "expected_patterns": ["Test:", "Input:", "Expected:"],
}
```

1. Tests automatically pick up new cases via `@pytest.mark.parametrize`.

## Handling Test Failures

### Structure Test Failures

If `test_output_contains_test_patterns` fails:

- Model may need more training on structured output format
- Check training data includes varied test case formats
- Consider lowering temperature for more deterministic output

### Keyword Test Failures

If `test_output_mentions_requirement_keywords` fails:

- Model may not be learning requirement-to-output mapping
- Review training data quality
- Check if keywords are too specific

### Consistency Test Failures

If `test_multiple_runs_pass_structure_check` fails:

- Model output is too variable
- Consider reducing temperature in generation
- May need more training epochs

## CI/CD Integration

Tests run automatically in the GitHub Actions inference workflow after model training completes. See `.github/workflows/inference.yml`.
