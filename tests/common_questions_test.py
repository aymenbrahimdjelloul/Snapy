"""
@author: Aymen Brahim Djelloul
version: 0.5
date: 28.06.2025
license: MIT License

An optimized test module for Snapy response generation with comprehensive testing capabilities.
"""

import re
import statistics
import time
import unittest
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from snapy import Snapy


@dataclass
class ExpectedResponse:
    """Data class for expected response patterns and validation criteria."""
    queries: List[str]
    expected_keywords: List[str]
    expected_patterns: List[str]
    min_confidence: float
    description: str
    max_response_time: float = 2.0  # seconds


class TestSnapyResponseGeneration(unittest.TestCase):
    """Comprehensive test cases for Snapy response generation."""

    @classmethod
    def setUpClass(cls):
        """Initialize test environment once for all tests."""
        cls.snapy = Snapy()
        cls.start_time = time.time()

        # Define expected responses with validation criteria
        cls.expected_responses = {
            "greeting": ExpectedResponse(
                queries=["Hello", "Hi", "Hey", "Good morning", "Greetings"],
                expected_keywords=["hello", "hi", "hey", "greetings", "nice", "good", "welcome"],
                expected_patterns=[r"hello", r"hi", r"how.*you", r"nice.*meet", r"good.*day"],
                min_confidence=0.7,
                description="Friendly greeting response"
            ),
            "math_simple": ExpectedResponse(
                queries=["What is 2+2?", "Calculate 5*3", "What's 10-7?"],
                expected_keywords=["4", "15", "3", "equals", "result", "answer"],
                expected_patterns=[r"\b4\b", r"\b15\b", r"\b3\b", r"answer.*4", r"result.*15"],
                min_confidence=0.8,
                description="Correct mathematical answers",
                max_response_time=1.5
            ),
            "personal_question": ExpectedResponse(
                queries=["How are you?", "How do you feel?", "Are you okay?"],
                expected_keywords=["good", "fine", "well", "ai", "assistant", "helping", "great"],
                expected_patterns=[r"i'm.*good", r"doing.*well", r"i.*ai", r"here.*help"],
                min_confidence=0.6,
                description="Appropriate AI status response"
            ),
            "weather": ExpectedResponse(
                queries=["What's the weather?", "How's the weather today?", "Is it raining?"],
                expected_keywords=["weather", "don't", "access", "check", "cannot", "real-time"],
                expected_patterns=[r"don't.*access", r"cannot.*provide", r"check.*weather", r"real.*time"],
                min_confidence=0.5,
                description="Weather data limitations explanation"
            ),
            "joke": ExpectedResponse(
                queries=["Tell me a joke", "Make me laugh", "Do you know any jokes?"],
                expected_keywords=["joke", "funny", "laugh", "why", "because", "pun"],
                expected_patterns=[r"why.*did", r"what.*do.*you.*call", r"knock.*knock", r"here's.*joke"],
                min_confidence=0.4,
                description="Humorous response attempt"
            )
        }

    @classmethod
    def tearDownClass(cls):
        """Print summary after all tests complete."""
        duration = time.time() - cls.start_time
        print(f"\nTest session completed in {duration:.2f} seconds")

    def _validate_response(self, result: Tuple[str, float], query: str = "") -> Tuple[str, float]:
        """Validate response format and content."""
        self.assertIsInstance(result, tuple, f"Expected tuple, got {type(result)}")
        self.assertEqual(len(result), 2, "Response should be (text, confidence) tuple")

        response, confidence = result
        self.assertIsInstance(response, str, "Response text should be string")
        self.assertIsInstance(confidence, (int, float), "Confidence should be numeric")
        self.assertGreater(len(response), 0, "Response should not be empty")
        self.assertTrue(0 <= confidence <= 1, "Confidence should be between 0 and 1")

        return response, confidence

    def _analyze_response(self, response: str, expected: ExpectedResponse) -> Dict:
        """Analyze response against expected patterns."""
        response_lower = response.lower()
        analysis = {
            "keyword_matches": [],
            "pattern_matches": [],
            "missing_keywords": [],
            "missing_patterns": []
        }

        # Check expected keywords
        for kw in expected.expected_keywords:
            if kw.lower() in response_lower:
                analysis["keyword_matches"].append(kw)
            else:
                analysis["missing_keywords"].append(kw)

        # Check expected patterns
        for pattern in expected.expected_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                analysis["pattern_matches"].append(pattern)
            else:
                analysis["missing_patterns"].append(pattern)

        # Calculate similarity score
        kw_score = len(analysis["keyword_matches"]) / max(len(expected.expected_keywords), 1)
        pattern_score = len(analysis["pattern_matches"]) / max(len(expected.expected_patterns), 1)
        analysis["similarity_score"] = (kw_score + pattern_score) / 2

        return analysis

    def _assess_quality(self, similarity: float, confidence: float, expected: ExpectedResponse) -> str:
        """Assess response quality based on similarity and confidence."""
        if similarity >= 0.8 and confidence >= expected.min_confidence:
            return "EXCELLENT"
        if similarity >= 0.6 and confidence >= expected.min_confidence * 0.9:
            return "GOOD"
        if similarity >= 0.4 or confidence >= expected.min_confidence * 0.7:
            return "FAIR"
        return "POOR"

    def test_response_validation(self):
        """Test response validation across all categories."""
        print("\nResponse Validation Tests:")
        results = []

        for category, expected in self.expected_responses.items():
            for query in expected.queries[:3]:  # Test first 3 queries per category
                try:
                    start_time = perf_counter()
                    result = self.snapy.generate_response(query)
                    response_time = perf_counter() - start_time

                    response, confidence = self._validate_response(result, query)
                    analysis = self._analyze_response(response, expected)
                    quality = self._assess_quality(analysis["similarity_score"], confidence, expected)

                    results.append({
                        "category": category,
                        "query": query,
                        "confidence": confidence,
                        "similarity": analysis["similarity_score"],
                        "quality": quality,
                        "time": response_time
                    })

                    print(f"{category:15} | {quality:8} | {confidence:.2f} | {query[:30]:30}...")

                    # Performance assertion
                    self.assertLessEqual(
                        response_time, expected.max_response_time,
                        f"Response time {response_time:.2f}s exceeds limit for {category}"
                    )

                except Exception as e:
                    self.fail(f"Failed {category} test: {str(e)}")

        # Print summary statistics
        if results:
            print("\nTest Summary:")
            avg_conf = statistics.mean(r["confidence"] for r in results)
            avg_sim = statistics.mean(r["similarity"] for r in results)
            avg_time = statistics.mean(r["time"] for r in results)

            print(f"Average Confidence: {avg_conf:.2f}")
            print(f"Average Similarity: {avg_sim:.2f}")
            print(f"Average Response Time: {avg_time:.3f}s")

    def test_edge_cases(self):
        """Test handling of edge case inputs."""
        edge_cases = [
            ("", "empty_input"),
            ("   ", "whitespace"),
            ("@#$%^&*", "special_chars"),
            ("A" * 500, "long_input"),
            ("123456", "numbers"),
            ("🤖🎉", "emojis"),
            ("Répétez", "unicode")
        ]

        print("\nEdge Case Tests:")
        for input_text, case_name in edge_cases:
            try:
                result = self.snapy.generate_response(input_text)
                response, confidence = self._validate_response(result, input_text)

                print(f"{case_name:15} | {confidence:.2f} | {response[:50]:50}...")

                # Basic assertions
                self.assertGreater(len(response), 0, "Should return some response")
                if case_name == "empty_input":
                    self.assertLess(confidence, 0.5, "Empty input should have low confidence")

            except ValueError as e:
                if case_name == "empty_input":
                    print("Empty input correctly raised ValueError")
                else:
                    self.fail(f"Unexpected error for {case_name}: {str(e)}")
            except Exception as e:
                self.fail(f"Error processing {case_name}: {str(e)}")

    def test_performance(self):
        """Test response generation performance."""
        test_queries = [
            "Hello",
            "What is 2+2?",
            "Explain quantum computing",
            "Tell me a joke",
            "What's the weather forecast?"
        ]

        print("\nPerformance Benchmark:")
        times = []

        for query in test_queries:
            start_time = perf_counter()
            result = self.snapy.generate_response(query)
            elapsed = perf_counter() - start_time

            self._validate_response(result, query)
            times.append(elapsed)

            print(f"{elapsed:.3f}s | {query[:50]:50}...")

        print(f"\nStats: Min={min(times):.3f}s, Avg={statistics.mean(times):.3f}s, Max={max(times):.3f}s")
        self.assertLess(max(times), 3.0, "Response time too slow")

    def test_confidence_distribution(self):
        """Test confidence score distribution across query types."""
        query_types = {
            "simple": ["Hello", "Hi there", "Good morning"],
            "factual": ["2+2", "Capital of France", "Square root of 16"],
            "complex": ["Explain quantum physics", "Discuss AI ethics"],
            "nonsense": ["asdfghjkl", "xyz123!@#", "foobar"]
        }

        print("\nConfidence Distribution:")
        confidence_scores = {q_type: [] for q_type in query_types}

        for q_type, queries in query_types.items():
            for query in queries:
                result = self.snapy.generate_response(query)
                _, confidence = self._validate_response(result, query)
                confidence_scores[q_type].append(confidence)

                print(f"{q_type:10} | {confidence:.2f} | {query[:30]:30}...")

        # Assert confidence trends
        self.assertGreater(
            statistics.mean(confidence_scores["simple"]),
            statistics.mean(confidence_scores["nonsense"]),
            "Simple queries should have higher confidence than nonsense"
        )


def interactive_test():
    """Interactive testing mode for manual verification."""
    print("\nInteractive Testing Mode (type 'quit' to exit)")
    snapy = Snapy()

    while True:
        try:
            query = input("\nEnter query: ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break

            if not query:
                print("Please enter a query")
                continue

            start_time = perf_counter()
            response, confidence = snapy.generate_response(query)
            elapsed = perf_counter() - start_time

            print(f"\nResponse ({elapsed:.3f}s, confidence={confidence:.2f}):")
            print(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    # Run automated tests
    unittest.main(argv=[''], verbosity=2, exit=False)

    # Launch interactive mode
    interactive_test()