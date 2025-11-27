from pathlib import Path
from dataclasses import dataclass, field
import json, os
import numpy as np
from matplotlib import pyplot as plt
from results.plots import retrieve_runs, parse_log, SFTResults
import code

PROJECT_ROOT = Path(__file__).parent.parent
logs_dir = PROJECT_ROOT / "logs"
results_dir = PROJECT_ROOT / "results"

def plot_test_nll_comparison(run_names, title="Test NLL Comparison", save_path=None, 
                             ylim=None, start_step=10, show=False):
    """
    Plot test NLL for multiple training runs on the same plot with log-scale x-axis.
    Similar to plot_nll_only but for multiple runs.
    
    Args:
        run_names: List of run name strings (e.g., ["gsm8k_sm1_16", "gsm8k_sm1_32", ...])
        title: Plot title
        save_path: Optional path to save the figure
        ylim: Optional tuple (min, max) for y-axis limits
        start_step: Minimum step to plot from (default 10)
        show: Whether to display the plot
    
    Example:
        runs = ["gsm8k_sm1_16", "gsm8k_sm1_32", "gsm8k_sm1_64"]
        plot_test_nll_comparison(runs, title="GSM8K SM1: Rank Comparison")
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Parse logs for all runs
    results_list = []
    for run_name in run_names:
        try:
            results = parse_log("sft", run_name)
            results_list.append((run_name, results))
        except Exception as e:
            print(f"Warning: Could not parse log for {run_name}: {e}")
            continue
    
    if len(results_list) == 0:
        raise ValueError("No valid runs found. Check run names and log files.")
    
    # Extract rank and LR from run names/config for sorting and labeling
    def extract_rank_lr(run_name, results_obj):
        """Extract rank and learning rate from run name or config."""
        rank = None
        lr = None
        
        # First, try to get from config (most reliable)
        if results_obj.config:
            rank = results_obj.config.get('lora_rank', None)
            lr = results_obj.config.get('learning_rate', None)
        
        # If not in config, try to parse from run name
        if rank is None or lr is None:
            parts = run_name.split('_')
            for i, part in enumerate(parts):
                try:
                    rank_val = int(part)
                    if i + 1 < len(parts):
                        try:
                            lr_val = float(parts[i + 1])
                            if rank is None:
                                rank = rank_val
                            if lr is None:
                                lr = lr_val
                            break
                        except ValueError:
                            pass
                    if rank is None:
                        rank = rank_val
                except ValueError:
                    pass
        
        return rank, lr
    
    # Sort runs by rank (and LR if rank is same)
    results_list.sort(key=lambda x: (extract_rank_lr(x[0], x[1])[0] or 0, 
                                      extract_rank_lr(x[0], x[1])[1] or 0))
    
    # Color scheme: from purple (low rank) to yellow (high rank)
    n_runs = len(results_list)
    colors = plt.cm.viridis(np.linspace(0, 1, n_runs))
    
    # Plot each run's test NLL
    for idx, (run_name, results) in enumerate(results_list):
        # Filter steps >= start_step
        test_iter_filtered = [s for s in results.test_iter if s >= start_step]
        test_nll_filtered = [nll for s, nll in zip(results.test_iter, results.test_mean_nll) 
                            if s >= start_step]
        
        if len(test_iter_filtered) == 0:
            print(f"Warning: No test data for {run_name} after step {start_step}")
            continue
        
        # Sort by step to ensure proper ordering
        sorted_data = sorted(zip(test_iter_filtered, test_nll_filtered))
        test_iter_filtered = [s for s, _ in sorted_data]
        test_nll_filtered = [nll for _, nll in sorted_data]
        
        # Create label with rank and LR info
        rank, lr = extract_rank_lr(run_name, results)
        if rank is not None and lr is not None:
            label = f"Rank {rank}, LR={lr:.0e}"
        elif rank is not None:
            label = f"Rank {rank}"
        elif lr is not None:
            label = f"LR={lr:.0e}"
        else:
            label = run_name
        
        ax.plot(test_iter_filtered, test_nll_filtered,
               color=colors[idx], label=label, linewidth=2, alpha=0.8)
    
    # Set log scale for x-axis (same as plot_nll_only)
    ax.set_xscale('log')
    ax.set_xlabel('Training Step (log scale)', fontsize=12)
    ax.set_ylabel('Test NLL', fontsize=12)
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Styling (same as plot_nll_only)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    
    return fig


def _collect_run_metadata(run_names):
    """Return list of (rank, lr, metrics, results)."""
    records = []
    for run_name in run_names:
        try:
            results = parse_log("sft", run_name)
        except Exception as exc:
            print(f"Skipping {run_name}: {exc}")
            continue
        cfg = results.config or {}
        rank = cfg.get("lora_rank")
        lr = cfg.get("learning_rate")
        if rank is None or lr is None:
            rank, lr = _infer_rank_lr_from_name(run_name, rank, lr)
        metrics = {
            "final_test_nll": results.test_mean_nll[-1] if results.test_mean_nll else np.nan,
        }
        records.append((run_name, rank, lr, metrics, results))
    if not records:
        raise ValueError("No valid runs with metadata found.")
    return records


def _summarize_rl_run(run_name: str) -> str:
    parts = run_name.split("_")

    approach = "Default"
    for token in parts:
        if "cipher" in token:
            approach = "Cipher"
            break
        if token.startswith("sm"):
            approach = "State Machine"
            break

    lr = parts[-2]
    clip = "w/ Clip" if "ppo" in run_name else "w/o Clip"

    return f"{approach}, {lr}, {clip}"


def _infer_rank_lr_from_name(run_name, rank, lr):
    parts = run_name.split("_")
    for idx, token in enumerate(parts):
        if rank is None:
            try:
                rank = int(token)
            except ValueError:
                pass
        if lr is None and idx + 1 < len(parts):
            try:
                lr = float(parts[idx + 1])
            except ValueError:
                pass
    return rank, lr


def _ema(values, alpha=0.2):
    """Simple exponential moving average."""
    if not values:
        return values
    smoothed = []
    acc = values[0]
    for val in values:
        acc = alpha * val + (1 - alpha) * acc
        smoothed.append(acc)
    return smoothed


def plot_rl_reward_sweep(
    run_names,
    title="GSM8K RL Reward Comparison",
    metric="train_reward",
    start_step=0,
    ema_alpha=0.2,
    show=False,
    save_path=None,
):
    """
    Plot RL reward/score traces for multiple runs on the same axes.
    """
    if not run_names:
        raise ValueError("run_names must contain at least one RL run.")

    metric_map = {
        "train_reward": ("train_iter", "train_mean_reward", "Train Mean Reward"),
        "train_score": ("train_iter", "train_mean_score", "Train Mean Score"),
        "test_score": ("test_iter", "test_mean_score", "Eval Mean Score"),
    }
    if metric not in metric_map:
        raise ValueError(f"metric must be one of {tuple(metric_map.keys())}")

    iter_attr, value_attr, ylabel = metric_map[metric]
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_names)))

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for color, run_name in zip(colors, run_names):
        results = parse_log("rl", run_name)
        steps = getattr(results, iter_attr, [])
        values = getattr(results, value_attr, [])
        if not steps or not values:
            continue

        filtered = [(s, v) for s, v in zip(steps, values) if s >= start_step]
        if not filtered:
            continue
        plot_steps, plot_values = map(list, zip(*filtered))

        if ema_alpha:
            plot_values = _ema(plot_values, alpha=ema_alpha)

        ax.plot(
            plot_steps,
            plot_values,
            label=_summarize_rl_run(run_name),
            color=color,
            linewidth=2,
            alpha=0.9,
        )
        plotted = True

    if not plotted:
        raise ValueError("No valid RL metrics found for provided runs.")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    header = plt.Line2D([], [], linestyle="", label="Reasoning Scheme, LR, Clipped Objective")
    ax.legend(
        [header, *handles],
        [header.get_label(), *labels],
        fontsize=9,
        loc="lower right",
        framealpha=0.9,
    )
    ax.set_ylim(0.2, 0.8)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_test_nll_sweep(run_names, title="GSM8k State Machine Compression", start_step=10, ylim=None, show=False, save_path=None, ema_alpha=0.2):
    """
    Plot test NLL curves for multiple runs on a single chart.
    Colors encode rank, line styles encode learning rate (similar to LoRA blog plots).
    An EMA is applied for smoother curves.
    """
    records = _collect_run_metadata(run_names)
    ranks = sorted({rec[1] for rec in records if rec[1] is not None})
    lrs = sorted({rec[2] for rec in records if rec[2] is not None})
    if not ranks or not lrs:
        raise ValueError("Could not infer ranks or learning rates from runs.")

    color_map = plt.cm.plasma(np.linspace(0.15, 0.85, len(ranks)))
    rank_to_color = {rank: color for rank, color in zip(ranks, color_map)}
    line_styles = ['--', '-', '-.']
    lr_to_style = {lr: line_styles[i % len(line_styles)] for i, lr in enumerate(lrs)}

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, rank, lr, _, results in records:
        steps = [s for s in results.test_iter if s >= start_step]
        nll = [n for s, n in zip(results.test_iter, results.test_mean_nll) if s >= start_step]
        if not steps:
            continue
        if ema_alpha:
            nll = _ema(nll, alpha=ema_alpha)
        label = f"{rank}, {lr:.0e}"
        ax.plot(
            steps,
            nll,
            linestyle=lr_to_style.get(lr, '-'),
            color=rank_to_color.get(rank, 'grey'),
            label=label,
            linewidth=2,
            alpha=0.9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training Examples", fontsize=12)
    ax.set_ylabel("Test NLL", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, which="both")
    handles, labels = ax.get_legend_handles_labels()
    parsed = []
    for lbl, h in zip(labels, handles):
        try:
            rank_str, lr_str = [tok.strip() for tok in lbl.split(",")]
            parsed.append((int(rank_str), float(lr_str), lbl, h))
        except ValueError:
            parsed.append((float("inf"), float("inf"), lbl, h))
    parsed.sort(key=lambda x: (x[0], x[1]))
    header = plt.Line2D([], [], linestyle="", label="Rank, learning rate")
    new_handles = [header] + [h for _, _, _, h in parsed]
    new_labels = ["Rank, learning rate"] + [lbl for _, _, lbl, _ in parsed]
    ax.legend(new_handles, new_labels, fontsize=10, loc="upper right", framealpha=0.9)
    ax.set_ylim(0.02, 1)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_final_test_nll_vs_lr(run_names, title="GSM8k State Machine Compression", save_path=None, show=False):
    records = _collect_run_metadata(run_names)
    rank_groups = {}
    ranks = sorted({rec[1] for rec in records if rec[1] is not None})
    color_map = plt.cm.plasma(np.linspace(0.15, 0.85, len(ranks))) if ranks else None
    rank_to_color = {rank: color for rank, color in zip(ranks, color_map)} if color_map is not None else {}

    for _, rank, lr, metrics, _ in records:
        rank_groups.setdefault(rank, []).append((lr, metrics["final_test_nll"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for rank, pairs in sorted(rank_groups.items()):
        pairs = sorted(pairs, key=lambda x: x[0])
        lrs = [p[0] for p in pairs]
        nlls = [p[1] for p in pairs]
        ax.plot(lrs, nlls, marker="o", label=f"Rank {rank}", color=rank_to_color.get(rank, None))
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Final Test NLL")
    ax.set_ylim(0.02, 0.1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    
    plot_sft = False
    plot_rl = True
    
    if plot_sft:
        runs = retrieve_runs("sft", "gsm8k_sm1_")
        plot_final_test_nll_vs_lr(runs, show=True, save_path=results_dir / "plots" / "gsm8k_sm1_ablation_final_nll.png")
        plot_test_nll_sweep(runs, show=True, save_path=results_dir / "plots" / "gsm8k_sm1_ablation_test_nll_sweep.png")

    if plot_rl:
        runs = retrieve_runs("rl", "SFT_gsm8k_gsm8k_")
        plot_rl_reward_sweep(runs, show=True, save_path=results_dir / "plots" / "gsm8k_rl_ablation_reward_sweep.png")