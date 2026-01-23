import json
import random

# We will generate synthetic "Product Requirements" and corresponding "Test Cases".
# Since we want to run this locally without an external API, we will use templates.

files = {
    "train": "data/train.jsonl",
    "val": "data/val.jsonl"
}

features = ["Login", "Checkout", "Search", "Profile", "inventory_api", "payment_gateway"]
actions = ["Verify that", "Ensure that", "Check if"]
conditions = ["user is logged in", "cart is empty", "network is slow", "input is invalid"]

templates = [
    {
        "req": "The system must prevent users from checking out if the inventory count is less than the requested quantity.",
        "tests": [
            "Scenario: Inventory check success\nGiven product A has stock 5\nWhen user adds 2 of product A\nThen checkout is allowed",
            "Scenario: Inventory check failure\nGiven product A has stock 5\nWhen user adds 6 of product A\nThen checkout is blocked with error 'Insufficient Stock'"
        ]
    },
    {
        "req": "The API endpoint /api/v1/users should return 401 Unauthorized if the Bearer token is missing.",
        "tests": [
            "Test: Valid Token\nRequest: GET /api/v1/users Headers: {Authorization: Bearer valid_token}\nResponse: 200 OK",
            "Test: Missing Token\nRequest: GET /api/v1/users Headers: {}\nResponse: 401 Unauthorized"
        ]
    },
    {
        "req": "Password fields must require at least one special character and be 8 chars long.",
        "tests": [
            "Test: Weak Password (No Special Char)\nInput: 'password123'\nExpected: Error 'Must contain special character'",
            "Test: Short Password\nInput: 'pass!'\nExpected: Error 'Must be 8+ characters'",
            "Test: Valid Password\nInput: 'Password123!'\nExpected: Success"
        ]
    }
]

def generate_entry():
    # In a real scenario, we'd have hundreds of unique inputs.
    # Here we rotate through our manual highly-quality templates to mimic the structure.
    base = random.choice(templates)
    return {
        "instruction": f"Generate QA test cases for the following requirement: {base['req']}",
        "input": "",
        "output": "\n\n".join(base['tests'])
    }

def main():
    print("Generating dataset...")
    
    # Generate 50 training examples (duplicated/randomized slightly for volume simulation)
    with open(files["train"], "w") as f:
        for _ in range(50):
            entry = generate_entry()
            f.write(json.dumps(entry) + "\n")
            
    # Generate 10 validation examples
    with open(files["val"], "w") as f:
        for _ in range(10):
            entry = generate_entry()
            f.write(json.dumps(entry) + "\n")
            
    print(f"Created {files['train']} (50 examples)")
    print(f"Created {files['val']} (10 examples)")

if __name__ == "__main__":
    main()
