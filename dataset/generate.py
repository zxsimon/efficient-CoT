from datasets import load_dataset
from common.utils import split_gsm8k
import json, argparse
from openai import OpenAI
from tqdm import tqdm

args = argparse.ArgumentParser()
args.add_argument("--approach", type=str, default="all", help="Approach to use for generation", choices=["all", "math_compress"])
args.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to use. Options: gsm8k", choices=["gsm8k"])
args = args.parse_args()

# Load prompts
with open("prompts.jsonl", "r") as f:
    prompts = {prompt["approach"]: prompt["prompt"] for prompt in [json.loads(line) for line in f.readlines()]}
approaches = list(prompts.keys()) if "all" in args.approach else args.approach

# ------------- Functions -------------

def load_and_process_dataset(ds_name, split = 'train', max_examples = 0):
    
    if ds_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")[split]
        process_fn = split_gsm8k
    else:
        raise ValueError(f"Dataset {ds_name} not currently supported")

    ds = ds.map(process_fn)
    if max_examples:
        ds = ds.select(range(max_examples))
    ds.name = ds_name

    return ds

def generate_completion(prompt, temperature = 0.0, max_tokens = 1000):
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
    model_name = client.models.list().data[0].id
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        res =  response.choices[0].message.content
    except Exception as e:
        print(f"Error generating completion: {e}. Prompt: {prompt}")
        res = None
    
    return res

BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
END = "\033[0m"

def print_comparison(question, answer, original_reasoning, transformed_reasoning):
    print(f"{BLUE}Question: {question}.\nAnswer: {answer}{END}")
    print(f"{RED}Original reasoning: {original_reasoning}{END}")
    print(f"{GREEN}Transformed reasoning: {transformed_reasoning}{END}")
    print("-"*100)

def transform_dataset(ds, approach, print_to_console = 5, overwrite = False):

    filename = f"{ds.name}_{approach}.jsonl"
    if overwrite:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('')

    with open(filename, 'a', encoding='utf-8') as f:
        for example in tqdm(ds, total = len(ds), desc = f"Transforming {ds.name} with {approach}"):
            question = example['question']
            reasoning = example['reasoning']
            answer = example['answer']
            prompt = prompts[approach].format(QUESTION=question, REASONING=reasoning, ANSWER=answer)
            completion = generate_completion(prompt)
            if completion:
                f.write(json.dumps({'question': question, 'answer': answer, 'original_reasoning': reasoning, 'transformed_reasoning': completion}) + '\n')    
                if print_to_console > 0:
                    print_to_console -= 1
                    print_comparison(question, answer, reasoning, completion)

if __name__ == "__main__":
    
    for approach in approaches:
        ds = load_and_process_dataset(args.dataset)
        transform_dataset(ds, approach, print_to_console = 10)