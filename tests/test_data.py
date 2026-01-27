"""
Golden test cases for model validation.

Each test case includes:
- instruction: The input requirement
- expected_keywords: Terms that should appear in the output
- expected_patterns: Regex patterns the output should match
- is_novel: If True, this requirement was NOT in training data (tests generalization)

All test cases use BDD (Given/When/Then) format since the model
is trained exclusively on BDD-formatted examples.
"""

# Test cases based on training data (tests learned patterns)
TRAINING_TEST_CASES = [
    {
        "id": "password_validation",
        "instruction": "Generate QA test cases for the following requirement: Password fields must require at least one special character and be 8 chars long.",
        "expected_keywords": ["password", "special", "character", "8", "given", "when", "then"],
        "expected_patterns": ["Scenario:", "Given", "When", "Then"],
        "is_novel": False,
    },
    {
        "id": "inventory_checkout",
        "instruction": "Generate QA test cases for the following requirement: The system must prevent users from checking out if the inventory count is less than the requested quantity.",
        "expected_keywords": ["inventory", "checkout", "quantity", "given", "when", "then"],
        "expected_patterns": ["Scenario:", "Given", "When", "Then"],
        "is_novel": False,
    },
    {
        "id": "api_authorization",
        "instruction": "Generate QA test cases for the following requirement: The API endpoint /api/v1/users should return 401 Unauthorized if the Bearer token is missing.",
        "expected_keywords": ["401", "token", "unauthorized", "given", "when", "then"],
        "expected_patterns": ["Scenario:", "Given", "When", "Then"],
        "is_novel": False,
    },
]

# Novel test cases NOT seen in training (tests generalization)
NOVEL_TEST_CASES = [
    {
        "id": "novel_refund_policy",
        "instruction": "Generate QA test cases for the following requirement: Refund requests must be submitted within 30 days of purchase and require order confirmation.",
        "expected_keywords": ["refund", "30", "days", "order", "given", "when", "then"],
        "expected_patterns": ["Scenario:", "Given", "When", "Then"],
        "is_novel": True,
    },
    {
        "id": "novel_geolocation",
        "instruction": "Generate QA test cases for the following requirement: The system must request location permission before displaying nearby stores on the map.",
        "expected_keywords": ["location", "permission", "stores", "map", "given", "when", "then"],
        "expected_patterns": ["Scenario:", "Given", "When", "Then"],
        "is_novel": True,
    },
    {
        "id": "novel_subscription_cancel",
        "instruction": "Generate QA test cases for the following requirement: Users must be able to cancel their subscription at any time and retain access until the billing period ends.",
        "expected_keywords": ["cancel", "subscription", "access", "billing", "given", "when", "then"],
        "expected_patterns": ["Scenario:", "Given", "When", "Then"],
        "is_novel": True,
    },
]

# Combined list for backward compatibility
TEST_CASES = TRAINING_TEST_CASES + NOVEL_TEST_CASES

# Patterns that indicate a valid test case structure (BDD format only)
VALID_STRUCTURE_PATTERNS = [
    r"Scenario:\s*\w+",          # Scenario: Something
    r"(Given|When|Then)\s+",     # BDD keywords
]

# Minimum and maximum output lengths (in characters)
MIN_OUTPUT_LENGTH = 50
MAX_OUTPUT_LENGTH = 2000

# Consistency test configuration
CONSISTENCY_RUNS = 5
CONSISTENCY_PASS_THRESHOLD = 0.8  # 80% of runs should pass
