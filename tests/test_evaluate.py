import unittest
import subprocess
import sys
import tempfile
from pathlib import Path
from evaluate import evaluate, markdown_report, percentile


def record(**kwargs):
    return dict({"id": "q1", "model": "fixture", "response": "Paris", "expected": "Paris", "latency_ms": 100, "cost_usd": 0}, **kwargs)


class EvaluationTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(evaluate([record(response=" PARIS  ")])["models"][0]["exact_match_rate"], 1)

    def test_error_case(self):
        result = evaluate([record(response="London")])
        self.assertEqual(result["models"][0]["exact_match_rate"], 0)
        self.assertIn('q1', markdown_report(result))

    def test_forbidden(self):
        result = evaluate([record(response=" SECRET token ", forbidden=["secret"])])
        self.assertEqual(result["models"][0]["forbidden_hit_rate"], 1)

    def test_duplicate(self):
        with self.assertRaises(ValueError):
            evaluate([record(), record()])

    def test_invalid_numbers(self):
        for value in (-1, float("nan"), float("inf"), True, "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate([record(latency_ms=value)])

    def test_empty(self):
        with self.assertRaises(ValueError):
            evaluate([])

    def test_percentile(self):
        self.assertAlmostEqual(percentile([100, 200], .95), 195)

    def test_mismatched_case_sets(self):
        result = evaluate([record(), record(id="q2", model="other")])
        self.assertFalse(result["comparable_case_sets"])
        self.assertIn("not a controlled comparison", markdown_report(result))

    def test_matched_case_sets(self):
        self.assertTrue(evaluate([record(), record(model="other")])["comparable_case_sets"])

    def test_invalid_forbidden(self):
        with self.assertRaises(ValueError):
            evaluate([record(forbidden=[""])])

    def test_cli_and_overwrite_protection(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            command = [sys.executable, str(root / "evaluate.py"), str(root / "examples/fixtures.jsonl"), "--output", str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = (output / "metrics.json").read_bytes()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual((output / "metrics.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
