# Command-line reference

Evaluate recorded responses into a new output directory:

```sh
python evaluate.py my-responses.jsonl --output reports/my-run
```

The destination must not already exist. This prevents accidental replacement of earlier results. Validation errors return exit code 2; successful evaluations return 0. Quality failures appear in the report and do not change the exit code.

To inspect the input format, see `examples/fixtures.jsonl`. Use the same case IDs and prompts for each model, keep generation settings fixed, and record costs and latency from your own application. The tool verifies matching IDs, not whether the underlying prompts or settings were identical.

No network traffic or model inference takes place. Reports include response text and expected answers in `metrics.json`; keep confidential records and generated reports out of public repositories.
