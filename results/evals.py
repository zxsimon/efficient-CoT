from results.plots import retrieve_runs
from pathlib import Path
from dataclasses import dataclass, field
import json, argparse
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
logs_dir = PROJECT_ROOT / "logs"

args = argparse.ArgumentParser()
args.add_argument("--prefix", type=str, default="eval_")
args = args.parse_args()

@dataclass
class evalResults:
    run_name: str = field(default_factory=str)
    scores: list = field(default_factory=list)
    total_evaluated: int = 0
    reason_lens: list = field(default_factory=list)

def parse_eval_results(run_jsonl):
    results = evalResults(run_name=run_jsonl)
    log_file = logs_dir / "eval" / run_jsonl
    if not log_file.exists():
        raise FileNotFoundError(f"Log file {log_file} not found")
    with open(log_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["type"] == "final":
                log = data["log"]
                results.scores = log["scores"]
                results.total_evaluated = log["total_evaluated"]
                results.reason_lens = log["reason_lens"]
    return results

def process_eval_results(results):
    results.scores = np.array(results.scores)
    results.total_evaluated = int(results.total_evaluated)
    results.reason_lens = np.array(results.reason_lens)
    print(f"Run name: {results.run_name}")
    print(f"Score mean: {results.scores.mean()}")
    print(f"Score standard error: {results.scores.std(ddof=1) / np.sqrt(results.total_evaluated)}")
    print(f"Score min: {results.scores.min()}")
    print(f"Score max: {results.scores.max()}")
    print(f"Total evaluated: {results.total_evaluated}")
    if results.reason_lens.size > 0:
        print(f"Reasoning length mean: {results.reason_lens.mean()}")
        print(f"Reasoning length standard error: {results.reason_lens.std(ddof=1) / np.sqrt(results.total_evaluated)}")
        print(f"Reasoning length min: {results.reason_lens.min()}")
        print(f"Reasoning length max: {results.reason_lens.max()}")
    else:
        print("No reasoning length data")
    print("-" * 100)
    return results

if __name__ == "__main__":
    runs = retrieve_runs("eval", args.prefix, ".jsonl")
    for run in runs:
        results = parse_eval_results(run)
        process_eval_results(results)