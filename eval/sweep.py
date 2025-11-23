import json, subprocess, shlex, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="eval/sweep_configs.jsonl")
args = parser.parse_args()

if "/" not in args.config:
    args.config = f"eval/{args.config}"
if not args.config.endswith(".jsonl"):
    args.config = f"{args.config}.jsonl"
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Config file {args.config} not found")

def main():

    with open(args.config) as f:
        for line in f:
            cfg = json.loads(line)
            args_str = " ".join(f"--{k} {v}" if v else f"--{k}" for k, v in cfg.items())
            cmd = f"python -m eval.eval {args_str}"
            print(cmd)
            subprocess.run(shlex.split(cmd), check=True)

if __name__ == "__main__":
    main()