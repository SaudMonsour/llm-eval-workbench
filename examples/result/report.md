# Recorded response evaluation

Metrics describe supplied records; this tool does not call or verify a model.

| Model | Cases | Exact match | Forbidden hits | p50 ms | p95 ms | Cost USD |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixture-a | 2 | 50.0% | 0.0% | 150.00 | 195.00 | 0.000000 |
| fixture-b | 2 | 100.0% | 0.0% | 225.00 | 292.50 | 0.000000 |

## Failed cases

- ["fixture-a", "arithmetic"]: exact_match=False, forbidden_hits=0
