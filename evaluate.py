"""Evaluate recorded LLM responses without making network requests."""
import argparse
import json
import math
import statistics
from pathlib import Path


def normalize(text):
    return " ".join(text.casefold().split())


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def evaluate(rows):
    if not rows:
        raise ValueError("At least one response is required")
    groups, seen = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each response must be an object")
        for field in ("id", "model", "response", "expected"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        key = (row["model"], row["id"])
        if key in seen:
            raise ValueError(f"Duplicate model/case pair: {key}")
        seen.add(key)
        for field in ("latency_ms", "cost_usd"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be a finite nonnegative number")
        forbidden = row.get("forbidden", [])
        if not isinstance(forbidden, list) or any(not isinstance(x, str) or not x.strip() for x in forbidden):
            raise ValueError("forbidden must contain nonempty strings")
        passed = normalize(row["response"]) == normalize(row["expected"])
        hits = [x for x in forbidden if normalize(x) in normalize(row["response"])]
        groups.setdefault(row["model"], []).append(dict(row, exact_match=passed, forbidden_hits=hits))
    summaries = []
    for model, cases in sorted(groups.items()):
        latencies = [x["latency_ms"] for x in cases]
        summaries.append({
            "model": model, "cases": len(cases),
            "exact_match_rate": statistics.mean(x["exact_match"] for x in cases),
            "forbidden_hit_rate": statistics.mean(bool(x["forbidden_hits"]) for x in cases),
            "latency_p50_ms": percentile(latencies, .5),
            "latency_p95_ms": percentile(latencies, .95),
            "total_cost_usd": sum(x["cost_usd"] for x in cases),
            "case_ids": sorted(x["id"] for x in cases),
        })
    matched = all(x["case_ids"] == summaries[0]["case_ids"] for x in summaries)
    return {"comparable_case_sets": matched, "models": summaries,
            "cases": [x for cases in groups.values() for x in cases]}


def markdown_report(result):
    lines = ["# Recorded response evaluation", "",
             "Metrics describe supplied records; this tool does not call or verify a model.", "",
             "| Model | Cases | Exact match | Forbidden hits | p50 ms | p95 ms | Cost USD |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["models"]:
        name = row["model"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        lines.append(f"| {name} | {row['cases']} | {row['exact_match_rate']:.1%} | {row['forbidden_hit_rate']:.1%} | {row['latency_p50_ms']:.2f} | {row['latency_p95_ms']:.2f} | {row['total_cost_usd']:.6f} |")
    if not result["comparable_case_sets"]:
        lines += ["", "Warning: models have different case sets. Their averages are not a controlled comparison."]
    lines += ["", "## Failed cases", ""]
    for row in result["cases"]:
        if not row["exact_match"] or row["forbidden_hits"]:
            identity = json.dumps([row["model"], row["id"]], ensure_ascii=False)
            lines.append(f"- {identity}: exact_match={row['exact_match']}, forbidden_hits={len(row['forbidden_hits'])}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL response records")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    try:
        rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = evaluate(rows)
        # Refuse to overwrite a previous evaluation.
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        (args.output / "report.md").write_text(markdown_report(result), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
