from pathlib import Path
from datasets import load_dataset, Dataset
from utils.utils import split_gsm8k, Colors
import json, argparse
from openai import OpenAI
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
dataset_dir = PROJECT_ROOT / "dataset"

args = argparse.ArgumentParser()
args.add_argument("--approach", type=str, nargs='+', default=["all"])
args.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "drop"])
args.add_argument("--overwrite", action="store_true", default=False)
args.add_argument("--split", type=str, nargs='+', default=["train", "test"], choices=["train", "test"])
args = args.parse_args()

def load_prompts():
    with open(dataset_dir / "prompts.jsonl", "r") as f:
        approaches = [json.loads(line) for line in f.readlines()]
    try:
        if "all" in args.approach:
            return [approach for approach in approaches if (approach["dataset"] == args.dataset or approach["dataset"] == "all")]
        else:
            return [approach for approach in approaches if (approach["approach"] in args.approach and (approach["dataset"] == args.dataset or approach["dataset"] == "all"))]
    except:
        print(f"Error loading prompts: No prompts found for dataset {args.dataset} and approach {args.approach}")
        return []
    
def load_and_process_dataset(ds_name, split = 'train', max_examples = 0):
    
    if ds_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")[split]
        ds = ds.map(split_gsm8k)
    
    elif ds_name == "drop":
        file_path = dataset_dir / f"drop_{split}.jsonl"
        if not file_path.exists():
            raise ValueError(f"DROP dataset not found at {file_path}.")
        
        df = pd.read_json(file_path, lines=True)
        ds = Dataset.from_pandas(df)
    
    else:
        raise ValueError(f"Dataset {ds_name} not currently supported")

    if max_examples:
        ds = ds.select(range(max_examples))
    ds.name = ds_name

    return ds

def get_client():
    return OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

def generate_completion(prompt, temperature = 0.0, max_tokens = 1000, client = None):
    if client is None:
        client = get_client()
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

def print_comparison(question, answer, original_reasoning, transformed_reasoning):
    print(f"{Colors.BLUE}Question: {question}.\nAnswer: {answer}{Colors.END}")
    print(f"{Colors.RED}Original reasoning: {original_reasoning}{Colors.END}")
    print(f"{Colors.GREEN}Transformed reasoning: {transformed_reasoning}{Colors.END}")
    print("-"*100)

def transform_dataset(ds, approach_dict, split, print_to_console = 5, client = None):
    """ds needs to have 'question', 'answer', 'reasoning' keys"""

    filename = dataset_dir / f"{ds.name}_{approach_dict['approach']}_{split}.jsonl"
    if args.overwrite:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('')

    with open(filename, 'a', encoding='utf-8') as f:
        for example in tqdm(ds, total = len(ds), desc = f"Transforming {ds.name} with {approach_dict['approach']}"):
            question = example['question']
            reasoning = example['reasoning']
            answer = example['answer']
            prompt = approach_dict['prompt'].format(QUESTION=question, REASONING=reasoning, ANSWER=answer)
            completion = generate_completion(prompt, client = client)
            if completion:
                f.write(json.dumps({'question': question, 'answer': answer, 'original_reasoning': reasoning, 'reasoning': completion}) + '\n')    
                if print_to_console > 0:
                    print_to_console -= 1
                    print_comparison(question, answer, reasoning, completion)

def empty_reasoning(input_file):
    """Replace reasoning field with empty string and save with 'empty' in filename."""
    parts = input_file.split('_')
    parts[1] = 'empty'
    output_file = '_'.join(parts)
    output_file = dataset_dir / output_file
    input_file = dataset_dir / input_file
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            data['reasoning'] = ""
            f_out.write(json.dumps(data) + '\n')
    
    print(f"Created {output_file}")
    return output_file

if __name__ == "__main__":

    for approach in load_prompts():
        for split in args.split:
            ds = load_and_process_dataset(args.dataset, split = split)
            client = get_client()
            transform_dataset(ds, approach, split, print_to_console = 10, client = client)