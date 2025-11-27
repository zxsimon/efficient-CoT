from utils.utils import extract_numerical_answer, split_reasoning_text, count_thinking_tokens, split_gsm8k, Colors, GMS8K_SUFFIX, DROP_SUFFIX, f1_score, process_drop
from datasets import load_dataset
from pathlib import Path
import json, torch, time
from tinker import types

PROJECT_ROOT = Path(__file__).parent.parent
dataset_dir = PROJECT_ROOT / "dataset"

# ---- Dataloader ----

def dataset_iterator(dataset_name, split = "train", num_examples = 0):
    """Each example is a dictionary with at least 'question', 'answer', 'reasoning' keys."""

    assert split in ["train", "test", "sample"]
    
    # GSM8K
    if "gsm8k" in dataset_name:
        
        if dataset_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main")[split].shuffle(seed=42)
            ds = ds.map(split_gsm8k)
            if num_examples:
                ds = ds.select(range(num_examples))
            ds = ds.map(lambda ex: {**ex, 'question': ex['question'] + GMS8K_SUFFIX})
        
        else:
            if not (dataset_dir / f"{dataset_name}_{split}.jsonl").exists():
                raise FileNotFoundError(f"Dataset {dataset_name}_{split} not found in {dataset_dir}")

            with open(dataset_dir / f"{dataset_name}_{split}.jsonl", "r") as f:
                ds = [json.loads(line) for line in f]
            for example in ds:
                example['question'] += GMS8K_SUFFIX

    # DROP
    elif "drop" in dataset_name:

        if split in ["train", "test"]:
        
            if not (dataset_dir / f"{dataset_name}_{split}.jsonl").exists():
                raise FileNotFoundError(f"Dataset {dataset_name}_{split} not found in {dataset_dir}")

            with open(dataset_dir / f"{dataset_name}_{split}.jsonl", "r") as f:
                ds = [json.loads(line) for line in f]

            for example in ds:
                example['question'] += DROP_SUFFIX
            
        else:

            ds = load_dataset("ucinlp/drop")["validation"].shuffle(seed=42)
            ds = ds.map(process_drop)
            if num_examples:
                ds = ds.select(range(num_examples))
            ds = ds.map(lambda ex: {**ex, 'question': ex['question'] + DROP_SUFFIX})
            ds = ds.map(lambda example: {"reasoning": ""})
    
    else:
        raise ValueError(f"Dataset {dataset_name}_{split} not currently supported")
    
    def _iterator():
        yield from ds
    
    return len(ds), _iterator()

def convert_to_datum(example, tokenizer, reasoning_tag = "reasoning"):
    prompt = f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
    completion = f"<think>\n{example[reasoning_tag]}\n</think>\n{example['answer']}<|im_end|>\n"

    prompt_tokens = tokenizer.encode(prompt)
    prompt_weights = [0] * len(prompt_tokens)
    completion_tokens = tokenizer.encode(completion)
    completion_weights = [1] * len(completion_tokens)
 
    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights
 
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
 
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
    )

def datum_to_encoded_prompt(datum):
    idx = ((torch.tensor(datum.loss_fn_inputs["weights"].data, dtype = torch.int) - 1) * -1).sum().item() + 1
    tokens = datum.model_input.chunks[0].tokens[:idx]
    return types.ModelInput.from_ints(tokens)


# ---- Evaluators ----

def evaluator(dataset_name, step, config, training_client, tokenizer, test_loader):

    metrics = {}

    start_time = time.time()
    print(f"Sampling on test set at step {step}. Printing first {config.print_count} examples.")
    total_score = 0
    reasoning_lens = []
    futures = []

    
    try:
        sample_batch = [next(test_loader) for _ in range(config.sample_count)]
    except StopIteration:
        print("Reached end of dataset. Reloading dataset.")
        _, test_loader = dataset_iterator(config.dataset_name, "test")
        sample_batch = [next(test_loader) for _ in range(config.sample_count)]
    
    sampling_client = training_client.save_weights_and_get_sampling_client()
    params = types.SamplingParams(max_tokens=1024, temperature=0.0, stop="<|im_end|>")

    for i, example in enumerate(sample_batch):
        datum = convert_to_datum(example, tokenizer)
        prompt = datum_to_encoded_prompt(datum)
        future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        futures.append(future)
        
    for i, (example, future) in enumerate(zip(sample_batch, futures)):
    
        result = future.result()

        result_text = tokenizer.decode(result.sequences[0].tokens)
        _, pred = split_reasoning_text(result_text)
        
        if dataset_name == "gsm8k":
            pred = extract_numerical_answer(pred)
            answer = float(example['answer'].replace(',', ''))
            score = 1.0 if answer == pred else 0.0
        elif dataset_name == "drop":
            score = f1_score(pred, example['answer'])
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

        if i < config.print_count:
            print(f"{Colors.RED}Example {i + 1} out of {len(sample_batch)} -- Correct answer: {example['answer']} -- Predicted answer: {pred}{Colors.END}")
            print(f"{Colors.BLUE}Prompt: {tokenizer.decode(prompt.chunks[0].tokens)}{Colors.END}")
            print(f"{Colors.GREEN}Response: {result_text}{Colors.END}")
    
        total_score += score
        tokens_tensor = torch.tensor([result.sequences[0].tokens])
        reasoning_lens.append(count_thinking_tokens(tokens_tensor, tokenizer)[0].item())

    metrics.update(
        step=step,
        sample_count=len(sample_batch),
        sample_score=total_score,
        sample_reasoning_lens=reasoning_lens,
        time_total=time.time() - start_time,
    )
    
    return metrics, test_loader

# ---- RL Reward Functions ----

def get_reward(ds_type, result, answer_text, tokenizer, max_penalty=1, penalize_correct_only=True, max_reasoning_tokens=1024, f1_threshold=0.8):
    """Unified reward function"""
    
    if ds_type == "gsm8k":
        return get_gsm8k_reward(result, answer_text, tokenizer, max_penalty, penalize_correct_only, max_reasoning_tokens)
    elif ds_type == "drop":
        return get_drop_reward(result, answer_text, tokenizer, max_penalty, penalize_correct_only, max_reasoning_tokens, f1_threshold)
    else:
        raise ValueError(f"Dataset type {ds_type} not supported")

def get_gsm8k_reward(result, answer_text, tokenizer, max_penalty, penalize_correct_only, max_reasoning_tokens) -> float:
    """Penalize all: toggle to penalize incorrect predictions for reasoning length too"""
    
    try:
        result_text = tokenizer.decode(result.sequences[0].tokens)
        _, pred = split_reasoning_text(result_text)
        answer = float(answer_text.replace(',', ''))
        pred = extract_numerical_answer(pred)
        
        correct = pred == answer
        tokens_tensor = torch.tensor([result.sequences[0].tokens])
        reasoning_len = count_thinking_tokens(tokens_tensor, tokenizer)[0].item()
    
    except ValueError as e:
        print(f"Error getting GSM8K reward: {e}")
        return 0

    def penalize_correct(correct, reasoning_len):
        """
        Reward bounds:
        - Maximum: 1.0 (correct prediction, zero reasoning length)
        - Minimum correct: 1 - max_penalty
        - Minimum: 0.0 (incorrect prediction, or reasoning length too long)
        """
        penalty = max_penalty * min(reasoning_len, max_reasoning_tokens) / max_reasoning_tokens
        reward = (1 - penalty) if correct else 0
        return reward

    def penalize_all(correct, reasoning_len):
        """
        Unnormalized Reward bound: [-1 , 1]
        - Maximum: 1.0 (correct prediction, zero reasoning length penalty)
        - Minimum correct: 1 - max_penalty (correct prediction and max reasoning length penalty)
        - Minimum: -1 * max_penalty (incorrect prediction and max reasoning length penalty)

        Normalized to [0, 1] range, assuming max_penalty is 1.
        """
        penalty = max_penalty * min(reasoning_len, max_reasoning_tokens) / max_reasoning_tokens
        unnormalized_reward = (1 if correct else 0) - penalty
        reward = (unnormalized_reward + 1) / 2
        return reward

    reward = penalize_correct(correct, reasoning_len) if penalize_correct_only else penalize_all(correct, reasoning_len)
    
    result = {
        "reward": reward,
        "score": correct,
        "reasoning_len": int(reasoning_len),
    }
    
    return result

def get_drop_reward(result, answer_text, tokenizer, max_penalty, penalize_correct_only, max_reasoning_tokens, f1_threshold) -> float:
    """Penalize all: toggle to penalize incorrect predictions for reasoning length too"""
    
    try:
        result_text = tokenizer.decode(result.sequences[0].tokens)
        _, pred = split_reasoning_text(result_text)
        
        f1 = f1_score(pred, answer_text)
        correct = f1 >= f1_threshold
        
        tokens_tensor = torch.tensor([result.sequences[0].tokens])
        reasoning_len = count_thinking_tokens(tokens_tensor, tokenizer)[0].item()
    
    except ValueError as e:
        print(f"Error getting DROP reward: {e}")
        return 0

    def penalize_correct(f1: float, correct: bool, reasoning_len: int):
        """
        Reward bounds:
        - Maximum: 1.0 (f1 >= f1_threshold, zero reasoning length)
        - Minimum correct: 1 - max_penalty
        - Minimum: 0.0 (f1 < f1_threshold and reasoning length too long)
        """
        penalty = max_penalty * min(reasoning_len, max_reasoning_tokens) / max_reasoning_tokens
        reward = (f1 - penalty) if correct else 0
        return reward

    def penalize_all(f1: float, reasoning_len: int):
        """
        Unnormalized Reward bound: [-1 , 1]
        - Maximum: 1.0 (f1 >= f1_threshold, zero reasoning length penalty)
        - Minimum: -1 * max_penalty (f1 < f1_threshold and max reasoning length penalty)

        Normalized to [0, 1] range, assuming max_penalty is 1.
        """
        penalty = max_penalty * min(reasoning_len, max_reasoning_tokens) / max_reasoning_tokens
        unnormalized_reward = f1 - penalty
        reward = (unnormalized_reward + 1) / 2
        return reward

    reward = penalize_correct(f1, correct, reasoning_len) if penalize_correct_only else penalize_all(f1, reasoning_len)
    
    result = {
        "reward": reward,
        "score": f1,
        "reasoning_len": int(reasoning_len),
    }
    
    return result