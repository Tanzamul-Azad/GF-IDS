"""Validate the archived protocol-v2 evidence without datasets, torch or GPUs.

Checks completion, checkpoint selection, configuration identity and every
aggregate metric present in the dataset summaries. It does not rerun inference.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

GROUP_FIELDS = ("model", "partition", "alpha", "aggregator", "mu", "class_weight",
                "init_broadcast", "lr", "lr_decay", "rounds", "epochs", "batch_size", "clients")


def read(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)


def check(root):
    checked_runs = checked_metrics = 0
    datasets = {}
    for dataset in ("cic", "nfton"):
        directory = root / "runs_v2" / dataset
        groups = defaultdict(list)
        count = 0
        for run_dir in sorted(directory.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "config.json").exists():
                continue
            records = {name: read(run_dir / f"{name}.json")
                       for name in ("config", "done", "history", "results")}
            config_record, done, history, results = (
                records[k] for k in ("config", "done", "history", "results"))
            config = config_record["config"]
            assert config_record["run_id"] == results["run_id"] == run_dir.name, run_dir
            assert results["config"] == config, f"Config mismatch: {run_dir}"
            assert results["done"] == done, f"Completion mismatch: {run_dir}"
            assert config["dataset"] == dataset, run_dir
            assert done["rounds"] == config["rounds"] == len(history), run_dir
            assert [h["round"] for h in history] == list(range(1, len(history) + 1)), run_dir
            selected = max(history, key=lambda h: h["val_macro_f1"])
            assert done["best"]["round"] == selected["round"], f"Selection mismatch: {run_dir}"
            for metric in ("val_accuracy", "val_macro_f1", "val_loss", "val_benign_far"):
                assert close(done["best"][metric], selected[metric]), (run_dir, metric)
            for h in history:
                assert len(h["up_bytes_per_client"]) == config["clients"], run_dir
                assert h["down_bytes"] > 0 and all(v > 0 for v in h["up_bytes_per_client"]), run_dir
            assert config_record["code_sha256"] and config["data_sha256"], run_dir
            for split in ("best/holdout", "best/holdout_nonoverlap", "final/holdout"):
                for metric in ("accuracy", "macro_f1", "benign_far", "attack_miss_rate"):
                    assert 0 <= results[split][metric] <= 1, (run_dir, split, metric)
            groups[tuple(config.get(k) for k in GROUP_FIELDS)].append(results)
            count += 1
        summaries = read(directory / "summary.json")
        assert len(summaries) == len(groups), dataset
        for summary in summaries:
            key = tuple(summary.get(k) for k in GROUP_FIELDS)
            runs = sorted(groups[key], key=lambda r: r["config"]["seed"])
            seeds = [r["config"]["seed"] for r in runs]
            assert summary["seeds"] == seeds == [42, 43, 44], (dataset, key)
            assert summary["best_rounds"] == [r["done"]["best"]["round"] for r in runs]
            for key, value in summary.items():
                if not key.endswith("/mean"):
                    continue
                split, metric = key.rsplit("/", 1)[0].rsplit("/", 1)
                if not all(split in r and metric in r[split] for r in runs):
                    raise AssertionError((dataset, key, "missing metric"))
                values = [r[split][metric] for r in runs]
                assert close(value, statistics.mean(values)), (dataset, key)
                assert summary[f"{split}/{metric}/values"] == values, (dataset, key)
                assert close(summary[f"{split}/{metric}/sd"], statistics.stdev(values)), (dataset, key)
                checked_metrics += 1
        datasets[dataset] = {"runs": count, "conditions": len(groups)}
        checked_runs += count
    return {"datasets": datasets, "completed_and_scored_runs": checked_runs,
            "aggregate_metrics_checked": checked_metrics,
            "scope": "Saved-record consistency; no training or inference was rerun."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = check(args.root.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
