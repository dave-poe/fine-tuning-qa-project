"""
Model validation tests for fine-tuned QA test case generator.

These tests validate that the fine-tuned model produces outputs that:
1. Have proper test case structure
2. Mention relevant keywords from the requirement
3. Meet quality standards (length, no repetition)
4. Are consistent across multiple runs
"""
import re
import pytest
from collections import Counter

from .test_data import (
    TEST_CASES,
    TRAINING_TEST_CASES,
    NOVEL_TEST_CASES,
    VALID_STRUCTURE_PATTERNS,
    MIN_OUTPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    CONSISTENCY_RUNS,
    CONSISTENCY_PASS_THRESHOLD,
)


class TestOutputStructure:
    """Tests that validate output contains proper test case structure."""

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
    def test_output_contains_test_patterns(self, generate_output, test_case):
        """Validate output contains at least one valid test case pattern."""
        output = generate_output(test_case["instruction"])

        has_valid_pattern = any(
            re.search(pattern, output, re.IGNORECASE)
            for pattern in VALID_STRUCTURE_PATTERNS
        )

        assert has_valid_pattern, (
            f"Output missing test case structure.\n"
            f"Expected one of: {VALID_STRUCTURE_PATTERNS}\n"
            f"Got: {output[:300]}..."
        )

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
    def test_output_has_multiple_test_cases(self, generate_output, test_case):
        """Validate output contains multiple test cases (at least 2)."""
        output = generate_output(test_case["instruction"])

        # Count occurrences of test case indicators
        test_count = len(re.findall(r"(Test:|Scenario:)", output, re.IGNORECASE))

        assert test_count >= 2, (
            f"Expected at least 2 test cases, found {test_count}.\n"
            f"Output: {output[:300]}..."
        )


class TestKeywordRelevance:
    """Tests that validate output mentions relevant keywords."""

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
    def test_output_mentions_requirement_keywords(self, generate_output, test_case):
        """Validate output mentions at least some keywords from the requirement."""
        output = generate_output(test_case["instruction"])
        output_lower = output.lower()

        keywords = test_case["expected_keywords"]
        matched_keywords = [kw for kw in keywords if kw.lower() in output_lower]

        # At least 50% of keywords should be present
        min_matches = max(1, len(keywords) // 2)

        assert len(matched_keywords) >= min_matches, (
            f"Output missing expected keywords.\n"
            f"Expected at least {min_matches} of: {keywords}\n"
            f"Found: {matched_keywords}\n"
            f"Output: {output[:300]}..."
        )


class TestOutputQuality:
    """Tests that validate output quality metrics."""

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
    def test_output_minimum_length(self, generate_output, test_case):
        """Validate output meets minimum length requirement."""
        output = generate_output(test_case["instruction"])

        assert len(output) >= MIN_OUTPUT_LENGTH, (
            f"Output too short. Expected >= {MIN_OUTPUT_LENGTH} chars, "
            f"got {len(output)}.\nOutput: {output}"
        )

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
    def test_output_maximum_length(self, generate_output, test_case):
        """Validate output doesn't exceed maximum length (no runaway generation)."""
        output = generate_output(test_case["instruction"])

        assert len(output) <= MAX_OUTPUT_LENGTH, (
            f"Output too long. Expected <= {MAX_OUTPUT_LENGTH} chars, "
            f"got {len(output)}. Possible runaway generation."
        )

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
    def test_no_excessive_repetition(self, generate_output, test_case):
        """Validate output doesn't contain excessive repetition."""
        output = generate_output(test_case["instruction"])

        # Split into sentences/lines and check for duplicates
        lines = [line.strip() for line in output.split("\n") if line.strip()]

        if len(lines) < 3:
            return  # Not enough lines to check

        # Count line occurrences
        line_counts = Counter(lines)
        most_common_count = line_counts.most_common(1)[0][1] if line_counts else 0

        # No single line should appear more than 3 times
        assert most_common_count <= 3, (
            f"Excessive repetition detected. "
            f"A line appears {most_common_count} times.\n"
            f"Output: {output[:300]}..."
        )

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
    def test_output_not_empty_or_gibberish(self, generate_output, test_case):
        """Validate output is not empty and contains readable content."""
        output = generate_output(test_case["instruction"])

        # Check it's not just whitespace
        assert output.strip(), "Output is empty or only whitespace"

        # Check it contains actual words (at least 10 words)
        words = re.findall(r"\b\w+\b", output)
        assert len(words) >= 10, (
            f"Output appears to be gibberish or too short. "
            f"Found only {len(words)} words."
        )


class TestConsistency:
    """Tests that validate output consistency across multiple runs."""

    @pytest.mark.parametrize("test_case", TRAINING_TEST_CASES[:2], ids=lambda tc: tc["id"])
    def test_multiple_runs_pass_structure_check(self, generate_output, test_case):
        """Validate that multiple runs consistently produce valid structure."""
        passes = 0

        for _ in range(CONSISTENCY_RUNS):
            output = generate_output(test_case["instruction"])
            has_valid_pattern = any(
                re.search(pattern, output, re.IGNORECASE)
                for pattern in VALID_STRUCTURE_PATTERNS
            )
            if has_valid_pattern:
                passes += 1

        pass_rate = passes / CONSISTENCY_RUNS

        assert pass_rate >= CONSISTENCY_PASS_THRESHOLD, (
            f"Consistency check failed. "
            f"Only {passes}/{CONSISTENCY_RUNS} runs ({pass_rate:.0%}) "
            f"produced valid output. Expected >= {CONSISTENCY_PASS_THRESHOLD:.0%}."
        )


class TestGeneralization:
    """Tests that validate model generalizes to unseen requirements.
    
    These tests use requirements NOT present in training data to verify
    the model learned the BDD pattern rather than memorizing examples.
    """

    @pytest.mark.parametrize("test_case", NOVEL_TEST_CASES, ids=lambda tc: tc["id"])
    def test_novel_prompt_produces_bdd_structure(self, generate_output, test_case):
        """Validate novel prompts still produce valid BDD structure."""
        output = generate_output(test_case["instruction"])

        has_valid_pattern = any(
            re.search(pattern, output, re.IGNORECASE)
            for pattern in VALID_STRUCTURE_PATTERNS
        )

        assert has_valid_pattern, (
            f"[GENERALIZATION] Model failed on unseen requirement.\n"
            f"Requirement: {test_case['instruction'][:100]}...\n"
            f"Expected BDD structure (Given/When/Then).\n"
            f"Got: {output[:300]}..."
        )

    @pytest.mark.parametrize("test_case", NOVEL_TEST_CASES, ids=lambda tc: tc["id"])
    def test_novel_prompt_mentions_keywords(self, generate_output, test_case):
        """Validate novel prompts include relevant domain keywords."""
        output = generate_output(test_case["instruction"])
        output_lower = output.lower()

        keywords = test_case["expected_keywords"]
        matched_keywords = [kw for kw in keywords if kw.lower() in output_lower]

        # At least 40% of keywords for novel prompts (slightly relaxed)
        min_matches = max(1, int(len(keywords) * 0.4))

        assert len(matched_keywords) >= min_matches, (
            f"[GENERALIZATION] Novel prompt missing keywords.\n"
            f"Expected at least {min_matches} of: {keywords}\n"
            f"Found: {matched_keywords}\n"
            f"Output: {output[:300]}..."
        )

    @pytest.mark.parametrize("test_case", NOVEL_TEST_CASES, ids=lambda tc: tc["id"])
    def test_novel_prompt_has_multiple_scenarios(self, generate_output, test_case):
        """Validate novel prompts produce multiple test scenarios."""
        output = generate_output(test_case["instruction"])

        # Count occurrences of scenario indicators
        scenario_count = len(re.findall(r"(Test:|Scenario:)", output, re.IGNORECASE))

        assert scenario_count >= 2, (
            f"[GENERALIZATION] Novel prompt should produce multiple scenarios.\n"
            f"Expected at least 2, found {scenario_count}.\n"
            f"Output: {output[:300]}..."
        )
