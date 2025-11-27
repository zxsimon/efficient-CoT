import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from utils.utils import f1_score
from tqdm import tqdm
from dataset.generate import generate_completion, get_client
from utils.utils import process_drop
from datasets import load_dataset
from pathlib import Path
import json, re
import code

PROJECT_ROOT = Path(__file__).parent.parent
dataset_dir = PROJECT_ROOT / "dataset"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def create_drop_reasoning_dataset(restart_index = None, reset_file = False):
    
    ds = load_dataset("ucinlp/drop")['train'].shuffle(seed=42)
    ds = ds.map(process_drop)

    client = get_client()
    
    with open(dataset_dir / "drop_reasoning.jsonl", 'w' if reset_file else 'a') as f:
        for i, example in tqdm(enumerate(ds), total=len(ds), desc="Processing DROP"):
            
            if restart_index is not None and i < restart_index:
                continue    

            context_id = example['context_id']
            question = example['question']
            answer = example['answer']
            answer_type = example['answer_type']
            prompt = f"{question} Provide your reasoning first. Then provide a concise final answer in <answer> tags."
            completion = generate_completion(prompt, client=client)

            f1 = 0.0
            
            try:
                if completion:
                    pattern = r'<answer>(.*?)</answer>'
                    match = re.search(pattern, completion)
                    if match:
                        pred = match.group(1)
                        f1 = f1_score(pred, answer)

            except Exception as e:
                print(f"Error calculating f1 score: {e}. Completion: {completion}. Answer: {answer}")

            print(completion, f1)

            f.write(json.dumps({'question': question, 'answer': answer, 'answer_type': answer_type, 'id': context_id, 'completion': completion, 'f1': f1}) + '\n')

def extract_answer(completion):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, completion, re.DOTALL)
    assert match is not None
    return match.group(1).strip()

def curate_drop_reasoning_dataset(threshold=0.8, test_size=1300, seed=42):
    
    # Load and filter by F1 threshold
    df = pd.read_json(dataset_dir / "drop_reasoning.jsonl", lines=True)
    df = df[df['f1'] >= threshold].reset_index(drop=True)
    
    # Extract and clean data
    df['extracted_answer'] = df['completion'].apply(extract_answer)
    df['reasoning'] = df['completion'].str.split('<answer>').str[0].str.strip()
    df['reasoning'] = df['reasoning'].str.split('Final answer:').str[0].str.strip()
    df['reasoning'] = df['reasoning'].str.split('Final Answer:').str[0].str.strip()
    
    # Group by ID to ensure no contamination
    # Get unique IDs and count examples per ID
    id_counts = df.groupby('id').size().reset_index(name='count')
    unique_ids = id_counts['id'].values
    
    # Shuffle unique IDs
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(unique_ids)
    
    # Split IDs to get approximately test_size examples in test set
    # Iteratively add IDs to test set until we reach target size
    test_ids = []
    test_count = 0
    for id_ in shuffled_ids:
        count = id_counts[id_counts['id'] == id_]['count'].values[0]
        if test_count + count <= test_size:
            test_ids.append(id_)
            test_count += count
        elif test_count < test_size:
            # Add this ID to reach closer to target, even if it exceeds slightly
            test_ids.append(id_)
            test_count += count
            break
    
    # Remaining IDs go to train
    train_ids = [id_ for id_ in unique_ids if id_ not in test_ids]
    
    # Split dataframe
    df_test = df[df['id'].isin(test_ids)].reset_index(drop=True)
    df_train = df[df['id'].isin(train_ids)].reset_index(drop=True)
    
    # Drop completion column before saving
    df_train = df_train.drop(columns=['completion'])
    df_test = df_test.drop(columns=['completion'])
    
    # Save to files
    df_train.to_json(dataset_dir / "drop_train.jsonl", orient='records', lines=True)
    df_test.to_json(dataset_dir / "drop_test.jsonl", orient='records', lines=True)
    
    # Also save the full curated dataset (for reference)
    df.to_json(dataset_dir / "drop_curated.jsonl", orient='records', lines=True)
    
    print(f"Dataset split complete:")
    print(f"  Train: {len(df_train)} examples from {len(train_ids)} unique contexts")
    print(f"  Test:  {len(df_test)} examples from {len(test_ids)} unique contexts")
    print(f"  Requested test size: {test_size}, actual: {len(df_test)}")
    print(f"  Total: {len(df)} examples from {len(unique_ids)} unique contexts")
    
    return df_train, df_test

curate_drop_reasoning_dataset(threshold = 0.8)
# code.interact(local=dict(globals(), **locals()))
# create_drop_reasoning_dataset(13882)