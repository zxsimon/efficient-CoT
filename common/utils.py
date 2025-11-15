import torch, re, math

def extract_numerical_answer(completion):
    """
    A very messy regex implementation to extract the numerical answer from a completion.
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
    assert think_start in answer_text, f"Expected {think_start} in answer text, got {answer_text}"
    parts = answer_text.split(think_end)
    
    # No answer due to max reasoning length
    if len(parts) == 1:
        answer = None
    elif len(parts) == 2:
        answer = parts[1].strip()
    else:
        raise ValueError(f"Expected 1 or 2 parts in answer text, got {len(parts)}. Raw answer text: {answer_text}")
    
    reasoning = parts[0].strip().strip(think_start).strip()
    return reasoning, answer


def split_gsm8k(example):
    answer_text = example['answer']
    parts = answer_text.split("####")
    assert len(parts) == 2, f"Expected 2 parts in answer text, got {len(parts)}"
    reasoning = parts[0].strip()
    answer = parts[1].strip()
    example['reasoning'] = reasoning
    example['answer'] = answer
    return example