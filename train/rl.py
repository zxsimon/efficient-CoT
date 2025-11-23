import os, tinker, time, torch
from concurrent.futures import Future
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
from pathlib import Path
from utils.checkpoint_utils import get_last_checkpoint, save_checkpoint
from utils.log_utils import Logger
from train.common import dataset_iterator, convert_to_datum, datum_to_encoded_prompt, evaluator, get_reward
import code

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class RLConfig:
    log_dir_prefix: str = "logs/rl"
    sft_dir_prefix: str = "logs/sft"
    model: str = "Qwen/Qwen3-8B"
    dataset_name: str = "gsm8k"
    
    batch_size: int = 32
    group_size: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 3
    max_iters: int = 1000
    
    load_sft_model: str = "gsm8k_sm1_32"
    lora_rank: int = 32
    
    save_every: int = 100
    evaluate_every: int = 5
    sample_count: int = 20
    print_count: int = 4

    f1_threshold: float = 0.8
    max_penalty: float = 1
    penalize_correct_only: bool = True
    penalizer_max_reasoning_tokens: int = 100


# ---- Training ----

def train_rl(config, allow_resume = True):

    if "gsm8k" in config.dataset_name:
        ds_type = "gsm8k"
    elif "drop" in config.dataset_name:
        ds_type = "drop"
    else:
        raise ValueError(f"Evaluator is not configured for dataset {config.dataset_name}")

    # Run statistics
    overall_start_time = time.time()
    total_examples_trained = 0
    
    # Initialize logger and file paths
    if config.load_sft_model:
        run_name = f"SFT_{config.dataset_name}_{config.load_sft_model}"
    else:
        run_name = f"{config.dataset_name}_{config.lora_rank}"

    log_dir = PROJECT_ROOT / config.log_dir_prefix / run_name
    sft_dir = PROJECT_ROOT / config.sft_dir_prefix / config.load_sft_model
    logger = Logger(project_name="rl", run_name=run_name, log_dir=log_dir, force_reset=not allow_resume)
    logger.log("config", asdict(config))

    # Load tokenizer, optimizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    adam_params = tinker.AdamParams(learning_rate=config.learning_rate)

    # Load data
    train_len, train_loader = dataset_iterator(config.dataset_name, "train")
    test_len, test_loader = dataset_iterator(config.dataset_name, "test")
    iters_per_epoch = train_len // config.batch_size
    num_iters = config.num_epochs * iters_per_epoch
    if num_iters > config.max_iters:
        num_iters = config.max_iters
    print(f"Loaded {train_len} examples for training. 1 epoch will run for {iters_per_epoch} iterations.")
    print(f"Loaded {test_len} examples for testing.")

    # Load clients and params
    service_client = tinker.ServiceClient()
    sampling_params = types.SamplingParams(max_tokens=1024, stop="<|im_end|>")

    # Look for checkpoints
    resume_info = get_last_checkpoint(log_dir) if allow_resume else None

    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_step = resume_info["step"] + 1
        print(f"Checkpoint found. Resuming RL training from step {start_step}")
    elif config.load_sft_model:
        start_step = 0
        sft_checkpoint = get_last_checkpoint(sft_dir)
        if not sft_checkpoint:
            raise FileNotFoundError(f"SFT checkpoint not found in {sft_dir}")
        training_client = service_client.create_training_client_from_state(
            sft_checkpoint["state_path"]
        )
        print(f"SFT checkpoint found. Starting RL training on top of SFT-trained model from {sft_dir}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model, rank=config.lora_rank
        )
        start_step = 0
        print(f"Starting base model RL training.")
    
    print(f"Starting RL training from step {start_step}")
    
    # Main training loop
    for step in tqdm(range(start_step, num_iters), total=num_iters, desc="Training"):
        start_time = time.time()
        metrics = {}
        test_metrics = {}
        final_step = step == num_iters - 1

        # Save checkpoint
        if final_step or (step % config.save_every == 0 and step > 0):
            save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=log_dir,
                kind="state",
                loop_state={"step": step},
            )
        
        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        training_datums: list[types.Datum] = []
        batch_scores: list[int] = []
        batch_reasoning_lens: list[int] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_prompts: list[list[int]] = []

        # Training step
        try:
            batch = [next(train_loader) for _ in range(config.batch_size)]
        except StopIteration:
            print("Reached end of dataset. Reloading dataset.")
            _, train_loader = dataset_iterator(config.dataset_name, "train")
            batch = [next(train_loader) for _ in range(config.batch_size)]
        
        for example in batch:
            datum = convert_to_datum(example, tokenizer)
            model_input = datum_to_encoded_prompt(datum)
            prompt_tokens = model_input.to_ints()

            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                future = sampling_client.sample(prompt=model_input, sampling_params=sampling_params, num_samples=1)
                sample_futures.append(future)

            batch_futures.append(sample_futures)
            batch_prompts.append(prompt_tokens)

        batch_answers = [example["answer"] for example in batch]

        for sample_futures, prompt_tokens, answer in zip(
            batch_futures, batch_prompts, batch_answers
        ):
            group_rewards: list[float] = []
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)

                result = get_reward(
                    ds_type, 
                    sample_result, 
                    answer, 
                    tokenizer, 
                    max_penalty=config.max_penalty, 
                    penalize_correct_only=config.penalize_correct_only, 
                    f1_threshold=config.f1_threshold,
                    max_reasoning_tokens=config.penalizer_max_reasoning_tokens
                    )
                
                batch_scores.append(result["score"])
                batch_reasoning_lens.append(result["reasoning_len"])
                group_rewards.append(result["reward"])

            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            if all(advantage == 0.0 for advantage in advantages):
                continue

            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)
        
        # Training step: Forward-backward pass
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="importance_sampling"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()
        
        # Training step: Log metrics
        total_examples_trained += len(batch)
        
        metrics.update(
            step=step,
            num_examples=len(batch),
            learning_rate=config.learning_rate,
            train_mean_reward=sum(batch_rewards) / len(batch_rewards),
            train_mean_score=sum(batch_scores) / len(batch_scores),
            train_mean_reasoning_len=sum(batch_reasoning_lens) / len(batch_reasoning_lens),
            progress=step / num_iters,
            time_total=time.time() - start_time,
        )
        logger.log("train", metrics)
        
        # Evaluate on test set once in a while
        if final_step or step % config.evaluate_every == 0:
            
            test_metrics, test_loader = evaluator(ds_type, step, config, training_client, tokenizer, test_loader)
            logger.log("test", test_metrics)
        

    # Save final checkpoint
    save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=log_dir,
        kind="both",
        loop_state={"step": step},
    )    
    
    print("Training Completed.")
    print(f"Total Training time: {time.time() - overall_start_time:.0f} seconds. Average time per iteration: {(time.time() - overall_start_time) / num_iters:.0f} seconds")
    print(f"Final step ({step}) training metrics: {metrics}")
    print(f"Final test metrics: {test_metrics}")
    print(f"Total examples trained: {total_examples_trained}")


config = RLConfig()
tokenizer = AutoTokenizer.from_pretrained(config.model)

if __name__ == "__main__":
    train_rl(config, allow_resume=False)