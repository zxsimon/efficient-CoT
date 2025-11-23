from collections import Counter
import torch, re, math
import tinker
from dataclasses import dataclass

GMS8K_SUFFIX = " Provide ONLY a final numerical answer, with no explanation, no units, no punctuation, and no other text."

DROP_SUFFIX = " Provide ONLY a concise final answer, with no explanation, no punctuation, and no other text."

@dataclass
class Colors:
    BLUE: str = "\033[94m"
    GREEN: str = "\033[92m"
    RED: str = "\033[91m"
    END: str = "\033[0m"

def extract_numerical_answer(completion):
    """
    A very messy regex implementation to extract the numerical answer from a completion. 
    Works fine for now, but needs to be further streamlined.
    In practice, shouldn't be an issue for larger models or post-SFT evaluation.
    """
    
    if not isinstance(completion, str):
        return float('nan')
    
    completion = completion.rstrip('.')
    def parse_number(num_str):
        """Convert string to int or float, removing commas."""
        try:
            cleaned = num_str.replace(',', '')
            return int(cleaned) if '.' not in cleaned else float(cleaned)
        except ValueError:
            print(f"Error parsing number: {num_str}")
            print(f"Full completion: {repr(completion)}")
            return float('nan')

    patterns = [
        r"\\boxed\{([-\d,\.]+)\}",                # LaTeX boxed: \boxed{42}
        r"####\s*([-\d,\.]+)",                    # GSM8K format: #### 42
        r"(?:answer is|answer:|=)\s*([-\d,\.]+)", # "answer is 42" or "= 42"
        r"^([-\d,\.]+)$",                         # Just a number
        r"([-\d,\.]+)\s*$",                       # Number at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            result = parse_number(match.group(1))
            if not math.isnan(result):
                return result
    
    # Fallback: find all valid numbers, must contain at least one digit
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", completion)
    if numbers:
        return parse_number(numbers[-1])
    
    return float('nan')

@torch.no_grad()
def count_thinking_tokens(batch_ids, tokenizer, think_start = "<think>", think_end = "</think>"):
    
    device = batch_ids.device
    think_start_id = tokenizer.encode(think_start, return_tensors="pt").squeeze().to(device)
    think_end_id = tokenizer.encode(think_end, return_tensors="pt").squeeze().to(device)
    max_token_length = batch_ids.shape[1] + 1 # include the last unfinished token

    start_idx = (batch_ids == think_start_id).long().argmax(dim = 1)
    end_idx = (batch_ids == think_end_id).long().argmax(dim = 1)
    
    # if end_idx is 0, then we reached max reasoning length without an answer
    end_idx = torch.where(end_idx == 0, max_token_length, end_idx)
    thinking_token_counts = end_idx - start_idx - 2 # exclude start and end tokens

    return thinking_token_counts
    

def split_reasoning_text(answer_text, think_start = "<think>", think_end = "</think>"):
    
    if think_start not in answer_text:
        print(f"WARNING: No {think_start} found in answer text: {answer_text}")
        return answer_text, None

    parts = answer_text.split(think_end)
    
    # No answer due to max reasoning length
    if len(parts) == 1:
        answer = None
    elif len(parts) == 2:
        answer = parts[1].strip().replace("<|im_end|>", "").strip()
    else:
        print(f"WARNING: Expected 1 or 2 parts in answer text, got {len(parts)}. Raw answer text: {answer_text}")
        return answer_text, None
    
    reasoning = parts[0].strip().strip(think_start).strip()

    return reasoning, answer


def normalize_f1_answer(text):
    
    if not isinstance(text, str):
        return text

    text = text.lower().strip()
    
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    
    text = re.sub(r'%$', '', text)
    text = re.sub(r'\$$', '', text)
    text = re.sub(r'\s*(percent|percentage)$', '', text)
    text = ' '.join(text.split())
    
    return text

def f1_score(completion, answer):

    if not isinstance(completion, str):
        return 0

    pred_tokens = normalize_f1_answer(completion).split()
    gt_tokens = normalize_f1_answer(answer).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def split_gsm8k(example):
    answer_text = example['answer']
    parts = answer_text.split("####")
    assert len(parts) == 2, f"Expected 2 parts in answer text, got {len(parts)}"
    reasoning = parts[0].strip()
    answer = parts[1].strip()
    example['reasoning'] = reasoning
    example['answer'] = answer
    return example

def process_drop(example):
    res = {}
    res['question'] = f"{example['passage']} {example['question']}"
    res['answer'] = example['answers_spans']['spans'][0]
    res['answer_type'] = example['answers_spans']['types'][0]
    res['context_id'] = example['section_id']
    return res


def compute_mean_nll(
    logprobs_list: list[tinker.TensorData], weights_list: list[tinker.TensorData]) -> float:
    """Taken from tinker-cookbook: https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/supervised/common.py"""
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list, strict=True):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()

    if total_weights == 0:
        print("WARNING: No valid weights found for NLL computation")
        return float("nan")

    return float(-total_weighted_logprobs / total_weights)

def calculate_dataset_reasoning_length(fp):
    # TODO: Implement this
    pass
    