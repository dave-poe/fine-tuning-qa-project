"""
Golden test cases for model validation.

Each test case includes:
- instruction: The input requirement
- expected_keywords: Terms that should appear in the output
- expected_patterns: Regex patterns the output should match
"""

TEST_CASES = [
    {
        "id": "password_validation",
        "instruction": "Generate QA test cases for the following requirement: Password fields must require at least one special character and be 8 chars long.",
        "expected_keywords": ["password", "special", "character", "8"],
        "expected_patterns": ["Test:", "Input:", "Expected:"],
    },
    {
        "id": "inventory_checkout",
        "instruction": "Generate QA test cases for the following requirement: The system must prevent users from checking out if the inventory count is less than the requested quantity.",
        "expected_keywords": ["inventory", "checkout", "quantity"],
        "expected_patterns": ["Scenario:", "Given", "When", "Then"],
    },
    {
        "id": "api_authorization",
        "instruction": "Generate QA test cases for the following requirement: The API endpoint /api/v1/users should return 401 Unauthorized if the Bearer token is missing.",
        "expected_keywords": ["401", "token", "unauthorized"],
        "expected_patterns": ["Test:", "Request:", "Response:"],
    },
    {
        "id": "email_validation",
        "instruction": "Generate QA test cases for the following requirement: Email fields must validate format and reject invalid email addresses.",
        "expected_keywords": ["email", "valid", "invalid"],
        "expected_patterns": ["Test:", "Input:", "Expected:"],
    },
    {
        "id": "session_timeout",
        "instruction": "Generate QA test cases for the following requirement: User sessions should expire after 30 minutes of inactivity.",
        "expected_keywords": ["session", "expire", "30", "minute"],
        "expected_patterns": ["Test:", "Scenario:"],
    },
]

# Patterns that indicate a valid test case structure
VALID_STRUCTURE_PATTERNS = [
    r"Test:\s*\w+",              # Test: Something
    r"Scenario:\s*\w+",          # Scenario: Something
    r"(Given|When|Then)\s+",     # BDD keywords
    r"Input:\s*",                # Input: ...
    r"Expected:\s*",             # Expected: ...
    r"Request:\s*",              # Request: ...
    r"Response:\s*",             # Response: ...
]

# Minimum and maximum output lengths (in characters)
MIN_OUTPUT_LENGTH = 50
MAX_OUTPUT_LENGTH = 2000

# Consistency test configuration
CONSISTENCY_RUNS = 5
CONSISTENCY_PASS_THRESHOLD = 0.6  # 60% of runs should pass
