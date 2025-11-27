from pathlib import Path
from dataclasses import dataclass, field
import json, os, argparse
import numpy as np
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
logs_dir = PROJECT_ROOT / "logs"
results_dir = PROJECT_ROOT / "results"

@dataclass
class SFTResults:
    config: dict = field(default_factory=dict)

    train_iter: list = field(default_factory=list)
    train_num_examples: list = field(default_factory=list)
    train_num_tokens: list = field(default_factory=list)
    train_learning_rate: list = field(default_factory=list)
    train_mean_nll: list = field(default_factory=list)
    train_time_total: list = field(default_factory=list)

    test_iter: list = field(default_factory=list)
    test_num_examples: list = field(default_factory=list)
    test_mean_nll: list = field(default_factory=list)
    test_time_total: list = field(default_factory=list)

    sample_iter: list = field(default_factory=list)
    sample_count: list = field(default_factory=list)
    sample_score: list = field(default_factory=list)
    sample_reasoning_lens: list = field(default_factory=list)
    sample_time_total: list = field(default_factory=list)

@dataclass
class RLResults:
    config: dict = field(default_factory=dict)

    train_iter: list = field(default_factory=list)
    train_num_examples: list = field(default_factory=list)
    train_learning_rate: list = field(default_factory=list)
    train_mean_reward: list = field(default_factory=list)
    train_mean_score: list = field(default_factory=list)
    train_mean_reasoning_len: list = field(default_factory=list)
    train_time_total: list = field(default_factory=list)

    test_iter: list = field(default_factory=list)
    test_num_examples: list = field(default_factory=list)
    test_mean_reward: list = field(default_factory=list)
    test_mean_score: list = field(default_factory=list)
    test_mean_reasoning_len: list = field(default_factory=list)
    test_time_total: list = field(default_factory=list)


def sft_run_title(results: SFTResults):
    config = results.config
    lr = config.get("learning_rate", 5e-4)
    rank = config.get("lora_rank")
    
    try:
        dataset, approach = config.get("dataset_name").split("_")
    except:
        dataset = config.get("dataset_name")
        approach = "Default"
    
    if "gsm8k" in dataset:
        dataset = "GSM8k"
    elif "drop" in dataset:
        dataset = "DROP"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    if "sm1" in approach:
        approach = "State Machine"
    elif "cipher1" in approach:
        approach = "Cipher"
    else:
        pass

    return f"{dataset} with {approach} reasoning. Rank {rank}, LR {lr}"


def rl_run_title(results: RLResults):
    config = results.config
    lr = config.get("learning_rate")
    rank = config.get("lora_rank")
    
    dataset = config.get("dataset_name")
    approach = config.get("load_sft_model").split("_")[1]
    
    if "gsm8k" in dataset:
        dataset = "GSM8k"
    elif "drop" in dataset:
        dataset = "DROP"
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    if "sm1" in approach:
        approach = "State Machine"
    elif "cipher1" in approach:
        approach = "Cipher"
    else:
        approach = "Default"

    return f"{dataset} with {approach} reasoning. Rank {rank}, LR {lr}"

def parse_log(run_type, run_name):
    
    log_file_name = f"{run_type}_{run_name}.jsonl"
    filepath = logs_dir / run_type / run_name / log_file_name

    assert filepath.exists(), f"Log file {filepath} not found"
    
    if run_type == "sft":
        results = SFTResults()

        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                log = data["log"]
                if data["type"] == "config":
                    results.config = log
                elif data["type"] == "train":
                    results.train_iter.append(log['step'])
                    results.train_num_examples.append(log['num_examples'])
                    results.train_num_tokens.append(log['num_tokens'])
                    results.train_learning_rate.append(log['learning_rate'])
                    results.train_mean_nll.append(log['train_mean_nll'])
                    results.train_time_total.append(log['time_total'])
                elif data["type"] == "test":
                    results.test_iter.append(log['step'])
                    results.test_num_examples.append(log['num_examples'])
                    results.test_mean_nll.append(log['test_mean_nll'])
                    results.test_time_total.append(log['time_total'])
                elif data["type"] == "sample":
                    results.sample_iter.append(log['step'])
                    results.sample_count.append(log['sample_count'])
                    results.sample_score.append(log['sample_score'])
                    results.sample_reasoning_lens.append(log['sample_reasoning_lens'])
                    results.sample_time_total.append(log['time_total'])
    
    elif run_type == "rl":
        results = RLResults()

        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                log = data["log"]
                if data["type"] == "config":
                    results.config = log
                elif data["type"] == "train":
                    results.train_iter.append(log['step'])
                    results.train_num_examples.append(log['num_examples'])
                    results.train_learning_rate.append(log['learning_rate'])
                    results.train_mean_reward.append(log['train_mean_reward'])
                    results.train_mean_score.append(log['train_mean_score'])
                    results.train_mean_reasoning_len.append(log['train_mean_reasoning_len'])
                    results.train_time_total.append(log['time_total'])
                elif data["type"] == "test":
                    results.test_iter.append(log['step'])
                    results.test_num_examples.append(log['sample_count'])
                    sample_count = log.get('sample_count', 1)
                    results.test_mean_score.append(log['sample_score'] / sample_count if sample_count > 0 else 0.0)
                    reasoning_lens = log.get('sample_reasoning_lens', [])
                    results.test_mean_reasoning_len.append(np.mean(reasoning_lens) if reasoning_lens else 0.0)
                    results.test_time_total.append(log['time_total'])
    else:
        raise ValueError(f"Invalid run type: {run_type}")
    
    return results


def plot_nll_only(results, run_name, save_path=None, nll_ylim=None, show=False):
    """
    Plot only train and test NLL with log scale x-axis.
    
    Args:
        results: SFTResults object
        run_name: Name of the run for the title
        save_path: Optional path to save the figure
        nll_ylim: Optional tuple of (min, max) for NLL y-axis limits. If None, auto-scale
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color scheme
    color_train = 'tab:blue'
    color_test = 'tab:orange'
    
    # Plot NLL
    ax.plot(results.train_iter, results.train_mean_nll, 
            color=color_train, label='Train NLL', linewidth=2, alpha=0.8)
    ax.plot(results.test_iter, results.test_mean_nll, 
            color=color_test, label='Test NLL', linewidth=2, alpha=0.8)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlabel('Training Step (log scale)', fontsize=12)
    ax.set_ylabel('Mean NLL', fontsize=12)
    
    # Set y-axis limits if provided
    if nll_ylim is not None:
        ax.set_ylim(nll_ylim)
    
    # Styling
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.title(f'Training NLL: {run_name}', fontsize=14, fontweight='bold')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    
    return fig


def plot_single_run_sft(results, run_name, save=True, nll_ylim=(0, 0.6), show=False, max_reasoning_len=200):
    """
    Plot SFT training results with 3 metrics on the same plot:
    1. Train vs Test mean NLL
    2. Sample correct percentage
    3. Mean reasoning length
    
    Args:
        results: SFTResults object
        run_name: Name of the run for the title
        save_path: Optional path to save the figure
        nll_ylim: Tuple of (min, max) for NLL y-axis limits. Default (0, 0.6)
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Color scheme
    color_train_nll = 'tab:blue'
    color_test_nll = 'tab:cyan'
    color_correct = 'tab:green'
    color_reasoning = 'tab:orange'
    
    # ============ Axis 1: Mean NLL (train and test) ============
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Mean NLL', color=color_train_nll, fontsize=12)
    
    line1 = ax1.plot(results.train_iter, results.train_mean_nll, 
                     color=color_train_nll, label='Train NLL', linewidth=2, alpha=0.8)
    line2 = ax1.plot(results.test_iter, results.test_mean_nll, 
                     color=color_test_nll, label='Test NLL', linewidth=2, 
                     marker='o', markersize=5, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_train_nll)
    ax1.set_ylim(nll_ylim)  # Set manual y-axis limits for NLL
    ax1.grid(True, alpha=0.3)
    
    # ============ Axis 2: Sample Correct Percentage ============
    ax2 = ax1.twinx()
    ax2.set_ylabel('Score Percentage (%)', color=color_correct, fontsize=12)
    
    # Compute correct percentage
    sample_score_pct = [100 * score / count for score, count 
                          in zip(results.sample_score, results.sample_count)]
    
    line3 = ax2.plot(results.sample_iter, sample_score_pct, 
                     color=color_correct, label='Score %', linewidth=2.5,
                     marker='s', markersize=6, alpha=0.9)
    ax2.tick_params(axis='y', labelcolor=color_correct)
    ax2.set_ylim(0, 105)  # Set limits for score percentage
    
    # ============ Axis 3: Mean Reasoning Length ============
    ax3 = ax1.twinx()
    # Offset the third axis to the right
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Mean Reasoning Length (tokens)', color=color_reasoning, fontsize=12)
    
    # Compute mean reasoning length for each step
    mean_reasoning_lens = [np.mean(lens) for lens in results.sample_reasoning_lens]
    
    line4 = ax3.plot(results.sample_iter, mean_reasoning_lens, 
                     color=color_reasoning, label='Mean Reasoning Length', linewidth=2.5,
                     marker='^', markersize=6, alpha=0.9, linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color_reasoning)
    ax3.set_ylim(0, max_reasoning_len)  # Set limits for reasoning length
    
    # ============ Title and Legend ============
    plt.title(f'{sft_run_title(results)}', fontsize=14, fontweight='bold', pad=20)
    
    # Combine all lines for a single legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    
    # Save or show
    if save:
        save_path = results_dir / "plots" / f"sft_{run_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    
    return fig


def plot_single_run_rl(results, run_name, save=True, reward_ylim=(0, 1.0), show=False, max_reasoning_len=200):
    """
    Plot RL training results with 3 metrics on the same plot:
    1. Train vs Test mean reward
    2. Score percentage
    3. Mean reasoning length
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # ============ Axis 1: Mean Reward (train and test) ============
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Mean Reward', color="#1f77b4", fontsize=12)  # blue for reward

    # Reward colors (metric = blue family)
    reward_color = "#1f77b4"  # blue

    line1 = ax1.plot(
        results.train_iter, results.train_mean_reward,
        color=reward_color, label='Train Reward',
        linewidth=2.0, linestyle='-',
        alpha=0.9,
    )[0]

    line2 = None
    if results.test_mean_reward and len(results.test_mean_reward) > 0:
        line2 = ax1.plot(
            results.test_iter, results.test_mean_reward,
            color=reward_color, label='Test Reward',
            linewidth=2.5, linestyle='--',
            marker='o', markersize=4,
            alpha=0.9,
        )[0]

    ax1.tick_params(axis='y', labelcolor=reward_color)
    ax1.set_ylim(reward_ylim)
    ax1.grid(True, alpha=0.3)

    # ============ Axis 2: Score Percentage ============
    ax2 = ax1.twinx()
    score_color = "#2ca02c"  # green
    ax2.set_ylabel('Score (%)', color=score_color, fontsize=12)

    train_score_pct = [100 * s for s in results.train_mean_score]
    test_score_pct = [100 * s for s in results.test_mean_score] if results.test_mean_score else []

    line3 = ax2.plot(
        results.train_iter, train_score_pct,
        color=score_color, label='Train Score %',
        linewidth=2.0, linestyle='-',
        alpha=0.8,
    )[0]

    line4 = None
    if results.test_iter and test_score_pct and len(results.test_iter) == len(test_score_pct):
        line4 = ax2.plot(
            results.test_iter, test_score_pct,
            color=score_color, label='Test Score %',
            linewidth=2.5, linestyle='--',
            marker='s', markersize=4,
            alpha=0.9,
        )[0]

    ax2.tick_params(axis='y', labelcolor=score_color)
    ax2.set_ylim(0, 105)

    # ============ Axis 3: Mean Reasoning Length ============
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))

    reason_color = "#ff7f0e"  # orange
    ax3.set_ylabel('Mean Reasoning Length (tokens)', color=reason_color, fontsize=12)

    line5 = ax3.plot(
        results.train_iter, results.train_mean_reasoning_len,
        color=reason_color, label='Train Reasoning Length',
        linewidth=2.0, linestyle='-',
        alpha=0.8,
    )[0]

    line6 = None
    if results.test_iter and results.test_mean_reasoning_len and len(results.test_iter) == len(results.test_mean_reasoning_len):
        line6 = ax3.plot(
            results.test_iter, results.test_mean_reasoning_len,
            color=reason_color, label='Test Reasoning Length',
            linewidth=2.5, linestyle='--',
            marker='^', markersize=4,
            alpha=0.9,
        )[0]

    ax3.tick_params(axis='y', labelcolor=reason_color)
    ax3.set_ylim(0, max_reasoning_len)

    # ============ Title and Legend ============
    plt.title(f'{rl_run_title(results)}', fontsize=14, fontweight='bold', pad=20)

    # Combine all lines for a single legend
    all_lines = [line1, line3, line5]
    if line2 is not None:
        all_lines.insert(1, line2)
    if line4 is not None:
        all_lines.append(line4)
    if line6 is not None:
        all_lines.append(line6)
    lines = all_lines
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.9)

    fig.tight_layout()

    if save:
        save_path = results_dir / "plots" / f"rl_{run_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()

    return fig



def retrieve_runs(run_type, prefix, suffix=None):
    runs = []
    for run in os.listdir(logs_dir / run_type):
        if "drop" in run or "gsm8k" in run:
            if run.startswith(prefix) and (suffix is None or run.endswith(suffix)):
                runs.append(run)
    return runs


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--sft", action="store_true", default=False)
    args.add_argument("--rl", action="store_true", default=False)
    args.add_argument("--save", action="store_true", default=False)
    args = args.parse_args()

    plot_sft = args.sft
    plot_rl = args.rl

    runs = retrieve_runs("sft", "",)
    if plot_sft:
        for run in runs:
            results = parse_log("sft", run)
            plot_single_run_sft(results, run, nll_ylim=(0, 0.6), save = args.save, max_reasoning_len = 500 if "drop" in run else 200)
            plot_nll_only(results, run, nll_ylim=(0, 0.6))


    runs = retrieve_runs("rl", "",)
    if plot_rl:
        for run in runs:
            results = parse_log("rl", run)
            plot_single_run_rl(results, run, reward_ylim=(0, 1.0), save = args.save, max_reasoning_len = 500 if "drop" in run else 200)


    plt.show()