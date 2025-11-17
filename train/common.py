from utils.utils import BLUE, GREEN, RED, END, extract_numerical_answer, split_reasoning_text, count_thinking_tokens, split_gsm8k
from datasets import load_dataset
from pathlib import Path
import json, tinker, torch, time
from tinker import types

PROJECT_ROOT = Path(__file__).parent.parent
dataset_dir = PROJECT_ROOT / "dataset"

# ---- Dataloader ----

def dataset_iterator(dataset_name, split = "train", num_examples = 0):
    """Each example is a dictionary with at least 'question', 'answer', 'reasoning' keys."""
    
    # GSM8K
    if "gsm8k" in dataset_name:
        
        gsm8k_suffix = " Provide ONLY a numerical answer without units, without any other text, explanation, or punctuation."

        if dataset_name == "gsm8k":
            ds = load_dataset("openai/gsm8k", "main")[split].shuffle(seed=42)
            ds = ds.map(split_gsm8k)
            if num_examples:
                ds = ds.select(range(num_examples))
            ds = ds.map(lambda ex: {**ex, 'question': ex['question'] + gsm8k_suffix})
        
        else:
            if not (dataset_dir / f"{dataset_name}.jsonl").exists():
                raise FileNotFoundError(f"Dataset {dataset_name} not found in {dataset_dir}")

            with open(dataset_dir / f"{dataset_name}.jsonl", "r") as f:
                ds = [json.loads(line) for line in f]
            for example in ds:
                example['question'] += gsm8k_suffix

    else:
        raise ValueError(f"Dataset {dataset_name} not currently supported")
    
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


# ---- Evaluator ----

def gsm8k_evaluator(step, config, training_client, tokenizer, test_loader):

    metrics = {}

    start_time = time.time()
    print(f"Sampling on test set at step {step}. Printing first {config.print_count} examples.")
    num_correct = 0
    reasoning_lens = []
    
    try:
        sample_batch = [next(test_loader) for _ in range(config.sample_count)]
    except StopIteration:
        print("Reached end of dataset. Reloading dataset.")
        _, test_loader = dataset_iterator(config.test_dataset_name, "test")
        sample_batch = [next(test_loader) for _ in range(config.sample_count)]
    sampling_client = training_client.save_weights_and_get_sampling_client()

    for i, example in enumerate(sample_batch):
        datum = convert_to_datum(example, tokenizer)
        prompt = datum_to_encoded_prompt(datum)
        params = types.SamplingParams(max_tokens=1024, temperature=0.0)
        future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
        result = future.result()

        result_text = tokenizer.decode(result.sequences[0].tokens)
        _, pred = split_reasoning_text(result_text)
        pred = extract_numerical_answer(pred)
        answer = float(example['answer'].replace(',', ''))

        if i < config.print_count:
            print(f"{RED}Example {i + 1} out of {len(sample_batch)} -- Correct answer: {example['answer']} -- Predicted answer: {pred}{END}")
            print(f"{BLUE}Prompt: {tokenizer.decode(prompt.chunks[0].tokens)}{END}")
            print(f"{GREEN}Response: {result_text}{END}")
        
        num_correct += answer == pred
        tokens_tensor = torch.tensor([result.sequences[0].tokens])
        reasoning_lens.append(count_thinking_tokens(tokens_tensor, tokenizer)[0].item())

    metrics.update(
        step=step,
        sample_count=len(sample_batch),
        sample_correct=num_correct,
        sample_reasoning_lens=reasoning_lens,
        time_total=time.time() - start_time,
    )
    
    return metrics, test_loader


# ---- RL Reward Functions ----

def get_gsm8k_reward(result, answer_text, tokenizer, penalizer = 0.0005, penalize_correct_only = True, max_reasoning_tokens = 1024) -> float:
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
        - Minimum correct: max(0, 1 - penalizer * max_reasoning_tokens)
        - Minimum: 0.0 (incorrect prediction, or reasoning length too long)
        """
        penalty = min(1, penalizer * min(reasoning_len, max_reasoning_tokens))
        reward = (1 - penalty) if correct else 0
        return reward

    def penalize_all(correct, reasoning_len):
        """
        Unnormalized Reward bound: [-1 , 1]
        - Maximum: 1.0 (correct prediction, zero reasoning length penalty)
        - Minimum correct: 0 (correct prediction and max reasoning length penalty)
        - Minimum: -1.0 (incorrect prediction and max reasoning length penalty)

        Normalized to [0, 1] range.
        """
        penalty = min(1, penalizer * min(reasoning_len, max_reasoning_tokens))
        reward = ((1 if correct else 0) - penalty + 1) / 2
        return reward

    reward = penalize_correct(correct, reasoning_len) if penalize_correct_only else penalize_all(correct, reasoning_len)
    
    result = {
        "reward": reward,
        "correct": correct,
        "reasoning_len": int(reasoning_len),
    }
    
    return result