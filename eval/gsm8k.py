from common.utils import split_gsm8k, split_reasoning_text, count_thinking_tokens, extract_numerical_answer
from itertools import islice
import datasets, torch
from tqdm import tqdm
import code

def eval_generator_gsm8k(split="test", max_examples=0):

    gsm8k = datasets.load_dataset("openai/gsm8k", "main")[split].shuffle(seed=42)
    
    if max_examples:
        try:
            gsm8k = gsm8k.select(range(max_examples))
        except IndexError:
            print(f"{max_examples} examples requested from GSM8K, but only {len(gsm8k)} examples available. Using all examples.")
    
    gsm8k = gsm8k.map(split_gsm8k)

    for example in gsm8k:
        question = example["question"]
        reasoning = example["reasoning"]
        answer = int(example["answer"].replace(',', ''))
        
        prompt = f"<|im_start|>user\nAnswer the following question:\n{question}\n"
        prompt += f"Output ONLY and IMMEDIATELY the final numerical answer and nothing else. Just the number. No other calculations, text, explanation, words, prefix, or punctuation."
        prompt += "<|im_end|>\n<|im_start|>assistant\n"

        yield prompt, answer, reasoning

@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, batch_size = 16, max_new_tokens = 1024, total_examples = 200, generator = None, reasoning = True, show = False):

    if generator is None:
        generator = eval_generator_gsm8k(max_examples=total_examples)

    total_evaluated = 0
    total_correct = 0
    num_iters = total_examples // batch_size
    reason_lens = torch.tensor([], device=model.device)

    for _ in tqdm(range(num_iters), total=num_iters, desc="Evaluating GSM8K"):

        try:
            prompts, answers, _ = map(list, zip(*islice(generator, batch_size)))
        except ValueError as e:
            print(f"Reached end of generator: {e}")
            break
        
        answers = torch.tensor(answers, dtype=torch.float32).to(model.device)
        
        # suppress thinking block
        if not reasoning:
            prompts = [prompt + "<think>\n\n</think>\n\n" for prompt in prompts]

        # tokenize
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=8, padding_side="left").to(model.device)
        input_ids, attention_mask = tokenized_prompts["input_ids"], tokenized_prompts["attention_mask"]

        # generate, decode, split reasoning and answer if necessary
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        new_ids = gen_ids[:, input_ids.shape[1]:]
        gen_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        
        if reasoning:
            gen_reasonings, gen_answers = zip(*[split_reasoning_text(gen_text) for gen_text in gen_texts])
        else:
            gen_answers = [gen_text.strip() for gen_text in gen_texts]

        # extract, evaluate, count
        processed_answers = torch.tensor([extract_numerical_answer(gen_answer) for gen_answer in gen_answers], dtype=torch.float32).to(model.device)
        num_correct = (processed_answers == answers).sum().item()
        total_correct += num_correct
        total_evaluated += batch_size
        
        if reasoning:
            reason_len = count_thinking_tokens(new_ids, tokenizer)
            reason_lens = torch.cat([reason_lens, reason_len])

        # debugging
        if show:
            if reasoning:
                print(f"Generated reasonings: {gen_reasonings}")
                print(f"Reasoning lengths: {reason_len}")
            print(f"Generated answers: {gen_answers}")
            print(f"Processed answers: {processed_answers}")
            print(f"Correct answer: {answers}")

    return total_correct, total_evaluated, reason_lens


if __name__ == "__main__":
    
    # Testing
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # model_name = "Qwen/Qwen3-8B"
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    if torch.cuda.is_available():
        model = torch.compile(model)
        torch.set_float32_matmul_precision('high')
    if model.device.type == "mps":
        model.set_attn_implementation("eager")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_correct, total_evaluated, reason_lens = evaluate_gsm8k(model, tokenizer, batch_size=2, total_examples=20, reasoning=False, show=True)
    code.interact(local=dict(globals(), **locals()))

    # print(evaluate_mmlu(model, tokenizer, batch_size=4, total_examples=100, show_modal_tokens=True))