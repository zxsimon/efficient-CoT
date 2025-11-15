from eval.gsm8k import evaluate_gsm8k
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, argparse

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
args.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k"])
args.add_argument("--batch_size", type=int, default=4)
args.add_argument("--total_examples", type=int, default=100)
args.add_argument("--reasoning", action="store_true", default=False)
args = args.parse_args()

# --------------------- TEMP OVERRIDES ---------------------
args.model = "Qwen/Qwen3-0.6B"
# --------------------- END TEMP OVERRIDES ---------------------


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model)
if model.device.type == "mps":
        model.set_attn_implementation("eager")


for reasoning in [False, True]:
    total_correct, total_evaluated, reason_lens = evaluate_gsm8k(model, tokenizer, batch_size=args.batch_size, total_examples=args.total_examples, reasoning=reasoning)
    print(f"Total correct: {total_correct}, Total evaluated: {total_evaluated}, Reasoning length: {reason_lens}")