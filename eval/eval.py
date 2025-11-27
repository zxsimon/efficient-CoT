from train.common import dataset_iterator
from utils.utils import split_reasoning_text, count_thinking_tokens, extract_numerical_answer, Colors, f1_score, normalize_f1_answer
from utils.log_utils import Logger
from tqdm import tqdm
from itertools import islice
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, argparse, os
from pathlib import Path

args = argparse.ArgumentParser()
args.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
args.add_argument("--checkpoint", type=str, default=None, help = "must exist in checkpoints/ e.g. gsm8k_sm1_32_0.0005")
args.add_argument("--no_reasoning", action="store_true", default=False)
args.add_argument("--batch_size", type=int, default=4)
args.add_argument("--total_examples", type=int, default=0)
args.add_argument("--show", action="store_true", default=False)
args.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "drop"])
args = args.parse_args()

PROJECT_ROOT = Path(__file__).parent.parent
checkpoints_dir = PROJECT_ROOT / "checkpoints"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def load_model():
    if args.checkpoint:
        return load_checkpoint(args.checkpoint)
    else:
        return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)

def load_checkpoint(checkpoint_name, base_model_name = "Qwen/Qwen3-8B"):
    # TODO: Figure out why Tinker does not save the base model name in the checkpoint adapter_config.json
    checkpoint_path = checkpoints_dir / checkpoint_name
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model_name, dtype=torch.bfloat16), checkpoint_path)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint {checkpoint_name}.")
    return model

def format_eval(example, reasoning = True):
    question = example["question"]
    answer = example["answer"]
    
    prompt = f"<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
    if not reasoning:
        prompt += "<think>\n\n</think>\n\n"
    
    return prompt, answer

@torch.no_grad()
def evaluate(model, tokenizer, dataset, batch_size = 16, max_new_tokens = 1024, total_examples = 0, reasoning = True, show = False, logger = None):

    if not reasoning:
        max_new_tokens = 20

    if dataset == "gsm8k":
        ds_len, generator = dataset_iterator("gsm8k", split="test", num_examples=total_examples)
    elif dataset == "drop":
        ds_len, generator = dataset_iterator("drop", split="sample", num_examples=total_examples)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    total_evaluated = 0
    scores = []
    num_iters = ds_len // batch_size
    reason_lens = torch.tensor([], device=model.device)


    for _ in tqdm(range(num_iters), total=num_iters, desc=f"Evaluating {dataset}"):

        try:
            batch = list(islice(generator, batch_size))
            formatted = [format_eval(example, reasoning) for example in batch]
            prompts, answers = map(list, zip(*formatted))
        except ValueError as e:
            print(f"Reached end of generator: {e}")
            break

        # tokenize
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=8, padding_side="left").to(model.device)
        input_ids, attention_mask = tokenized_prompts["input_ids"], tokenized_prompts["attention_mask"]

        # greedy generate, decode, split reasoning and answer if necessary
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        new_ids = gen_ids[:, input_ids.shape[1]:]
        gen_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        
        if reasoning:
            gen_reasonings, gen_answers = zip(*[split_reasoning_text(gen_text) for gen_text in gen_texts])
        else:
            gen_answers = [gen_text.strip() for gen_text in gen_texts]

        # extract, evaluate, count
        
        if dataset == "gsm8k":
            processed_answers = [extract_numerical_answer(gen_answer) for gen_answer in gen_answers]
            correct_answers = [float(answer.replace(',', '')) for answer in answers]
            batch_scores = [1.0 if processed_answer == correct_answer else 0.0 for processed_answer, correct_answer in zip(processed_answers, correct_answers)]
        elif dataset == "drop":
            batch_scores = [f1_score(gen_answer, answer) for gen_answer, answer in zip(gen_answers, answers)]
            processed_answers = [normalize_f1_answer(gen_answer) for gen_answer in gen_answers]
            correct_answers = [normalize_f1_answer(answer) for answer in answers]
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        
        scores.extend(batch_scores)
        total_evaluated += batch_size
        
        if reasoning:
            reason_len = count_thinking_tokens(new_ids, tokenizer)
            reason_lens = torch.cat([reason_lens, reason_len])

        if logger:
            for i, (prompt, gen_answer, processed_answer, correct_answer) in enumerate(zip(prompts, gen_answers, processed_answers, answers)):
                logger.log("eval", {
                    "prompt": prompt,
                    "gen_answer": gen_answer,
                    "processed_answer": processed_answer,
                    "correct_answer": correct_answer,
                    "score": batch_scores[i],
                    "reasoning": gen_reasonings[i] if reasoning else None,
                    "reasoning_len": reason_len[i].item() if reasoning else None,
                })
        
        # debugging
        if show:
            for i, (prompt, gen_answer, processed_answer, correct_answer) in enumerate(zip(prompts, gen_answers, processed_answers, answers)):
                print(f"\n--- Example {i+1} ---")
                print(f"{Colors.BLUE}Prompt: {prompt}{Colors.END}")
                print(f"{Colors.GREEN}Generated answer: {gen_answer}{Colors.END}")
                print(f"{Colors.RED}Processed answer: {processed_answer}{Colors.END}")
                print(f"{Colors.RED}Correct answer: {correct_answer}{Colors.END}")
                if reasoning:
                    print(f"{Colors.GREEN}Generated reasoning: {gen_reasonings[i]}{Colors.END}")
                    print(f"{Colors.GREEN}Reasoning length: {reason_len[i]}{Colors.END}")

    return scores, total_evaluated, reason_lens


if __name__ == "__main__":

    reasoning = not args.no_reasoning

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model()
    run_name=args.checkpoint if args.checkpoint else f"{args.dataset}_{args.model.split('/')[-1]}_{reasoning}"
    logger = Logger(project_name="eval", run_name=run_name, log_dir=PROJECT_ROOT / "logs" / "eval")
    
    print(f"Evaluating {args.checkpoint if args.checkpoint else args.model} on {args.dataset} with batch size {args.batch_size}.")
    print(f"Device: {model.device}")
    
    if model.device.type == "mps":
        model.set_attn_implementation("eager")
    if model.device.type == "cuda":
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')

    if args.checkpoint:
        if "gsm8k" in args.checkpoint:
            args.dataset = "gsm8k"
        elif "drop" in args.checkpoint:
            args.dataset = "drop"
        else:
            raise ValueError(f"Checkpoint {args.checkpoint} not supported")
    
    scores, total_evaluated, reason_lens = evaluate(
        model, 
        tokenizer, 
        dataset=args.dataset,
        batch_size=args.batch_size, 
        total_examples=args.total_examples, 
        reasoning=not args.no_reasoning, 
        show=args.show,
        logger=logger)

    logger.log("final", {
        "scores": scores,
        "total_evaluated": total_evaluated,
        "reason_lens": reason_lens.tolist(),
    })

    print(f"Evaluation completed for model: {args.checkpoint if args.checkpoint else args.model}.")
    print(f"Total evaluated: {total_evaluated}.")
    print(f"Pass@1 score: {sum(scores) / total_evaluated:.2f}")
    
    if not args.no_reasoning:
    
        reason_lens_mean = reason_lens.mean().item()
        reason_lens_std = reason_lens.std().item()
        reason_lens_min = reason_lens.min().item()
        reason_lens_max = reason_lens.max().item()
        print(f"Reasoning length: {reason_lens_mean:.0f} ± {reason_lens_std:.0f} (min: {reason_lens_min:.0f}, max: {reason_lens_max:.0f})")