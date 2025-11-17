import os, tinker, time
from tqdm import tqdm
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer
from pathlib import Path
from utils.utils import compute_mean_nll
from utils.checkpoint_utils import get_last_checkpoint, save_checkpoint
from utils.log_utils import Logger
from train.common import dataset_iterator, convert_to_datum, gsm8k_evaluator

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class SFTConfig:
    log_dir_prefix: str = "logs/sft"
    model: str = "Qwen/Qwen3-8B"
    train_dataset_name: str = "gsm8k"
    test_dataset_name: str = "gsm8k"
    
    batch_size: int = 16
    learning_rate: float = 5e-4
    num_epochs: int = 1
    max_iters: int = 10
    
    lora_rank: int = 32
    
    save_every: int = 100
    sample_every: int = 5
    sample_count: int = 10
    evaluate_every: int = 5
    evaluate_count: int = 16
    print_count: int = 4


# ---- Training ----


def train_sft(config, allow_resume = True):

    # Run statistics
    overall_start_time = time.time()
    total_tokens_trained = 0
    total_examples_trained = 0
    
    # Initialize logger
    run_name = f"{config.lora_rank}_{config.train_dataset_name}"
    log_dir = PROJECT_ROOT / config.log_dir_prefix / run_name
    logger = Logger(project_name="sft", run_name=run_name, log_dir=log_dir, force_reset=not allow_resume)
    logger.log("config", asdict(config))

    # Load tokenizer, optimizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    adam_params = tinker.AdamParams(learning_rate=config.learning_rate)

    # Load data
    train_len, train_loader = dataset_iterator(config.train_dataset_name, "train")
    test_len, test_loader = dataset_iterator(config.test_dataset_name, "test")
    iters_per_epoch = train_len // config.batch_size
    num_iters = config.num_epochs * iters_per_epoch
    if num_iters > config.max_iters:
        num_iters = config.max_iters
    print(f"Loaded {train_len} examples for training. 1 epoch will run for {iters_per_epoch} iterations.")
    print(f"Loaded {test_len} examples for testing.")

    # Load clients
    service_client = tinker.ServiceClient()

    # Check for resuming
    resume_info = get_last_checkpoint(log_dir) if allow_resume else None
    if resume_info:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_step = resume_info["step"]
        print(f"Resuming from step {start_step}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model, rank=config.lora_rank
        )
        start_step = 0
    
    print(f"Starting SFT training from step {start_step}")

    # Main training loop
    for step in tqdm(range(start_step, num_iters), total=num_iters, desc="Training"):
        start_time = time.time()
        metrics = {}
        sample_metrics = {}
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
        
        # Training step
        try:
            batch = [next(train_loader) for _ in range(config.batch_size)]
        except StopIteration:
            print("Reached end of dataset. Reloading dataset.")
            _, train_loader = dataset_iterator(config.train_dataset_name, "train")
            batch = [next(train_loader) for _ in range(config.batch_size)]
        converted_batch = [convert_to_datum(example, tokenizer) for example in batch]
        fwd_bwd_future = training_client.forward_backward(converted_batch, loss_fn="cross_entropy")
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Compute train metrics
        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in converted_batch]
        train_nll = compute_mean_nll(train_logprobs, train_weights)
        num_tokens = sum(d.model_input.length for d in converted_batch)
        
        # Log metrics
        total_examples_trained += len(batch)
        total_tokens_trained += num_tokens
        
        metrics.update(
            step=step,
            num_examples=len(batch),
            num_tokens=num_tokens,
            learning_rate=config.learning_rate,
            train_mean_nll=train_nll,
            progress=step / num_iters,
            time_total=time.time() - start_time,
        )
        logger.log("train", metrics)
        
        
        # Evaluate on test set once in a while
        if final_step or (step % config.evaluate_every == 0 and step > 0):
            start_time = time.time()
            print(f"Evaluating on test set at step {step}.")
            try:
                batch = [next(test_loader) for _ in range(config.evaluate_count)]
            except StopIteration:
                print("Reached end of dataset. Reloading dataset.")
                _, test_loader = dataset_iterator(config.test_dataset_name, "test")
                batch = [next(test_loader) for _ in range(config.evaluate_count)]
            converted_batch = [convert_to_datum(datum, tokenizer) for datum in batch]
            fwd_bwd_future = training_client.forward_backward(converted_batch, loss_fn="cross_entropy")
            fwd_bwd_result = fwd_bwd_future.result()

            # Compute test metrics
            test_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            test_weights = [d.loss_fn_inputs["weights"] for d in converted_batch]
            test_nll = compute_mean_nll(test_logprobs, test_weights)
            
            print(f"Test NLL at step {step}: {test_nll}")
            test_metrics.update(
                step=step,
                num_examples=len(batch),
                test_mean_nll=test_nll,
                time_total=time.time() - start_time,
            )
            logger.log("test", test_metrics)
        
        
        # Sample on test set once in a while
        if final_step or (step % config.sample_every == 0 and step > 0):
            
            sample_metrics, test_loader = gsm8k_evaluator(step, config, training_client, tokenizer, test_loader)
            logger.log("sample", sample_metrics)
        

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
    print(f"Final step ({step}) sample metrics: {sample_metrics}")
    print(f"Total tokens trained: {total_tokens_trained}. Total examples trained: {total_examples_trained}")


config = SFTConfig()
tokenizer = AutoTokenizer.from_pretrained(config.model)

if __name__ == "__main__":
    train_sft(config, allow_resume=False)