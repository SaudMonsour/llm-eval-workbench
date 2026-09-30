# LLM Eval Workbench

A small command-line tool for checking recorded LLM answers before changing a prompt or switching models.

## What it does

Read response records from JSONL and write a JSON result plus a Markdown report. Each report includes normalized exact-match accuracy, forbidden-phrase checks, median and p95 latency, supplied cost, and failed case IDs. Different case sets produce a comparison warning rather than a misleading ranking.

The first version runs entirely offline and uses only Python's standard library. It makes no API calls and needs no API key. Bring records from your own application; latency and cost are supplied measurements, not estimates made by this tool.

## Record format

Each line contains `id`, `model`, `response`, `expected`, `latency_ms`, and `cost_usd`. An optional `forbidden` list checks for unwanted phrases. IDs must be unique within each model. Numeric measurements must be finite and nonnegative.

## Current scope

This is an evaluation utility, not a trained model or a published LLM benchmark. The example records are handwritten fixtures, not real model outputs. They exercise both successful and failed answers; their timings are illustrative.

Exact match ignores case and repeated whitespace but keeps punctuation. It is useful for short factual answers and structured test cases, not open-ended writing. Phrase matching is a diagnostic check, not a security guarantee. p95 uses linear interpolation; small samples do not establish reliable production latency.

## Files

- `evaluate.py`: validation, scoring, and report generation.
- `tests/test_evaluate.py`: unit tests for scoring and rejected inputs.
- `examples/fixtures.jsonl`: clearly labeled example records.
- `examples/result/`: output from the fixture run.

Python 3.10 or newer is required. There are no third-party dependencies.

## Development

Created for Saud Alotaibi's AI engineering portfolio with AI-assisted implementation. The goal is a useful, inspectable tool rather than a benchmark claim. Live provider adapters and semantic scoring are not implemented yet.
