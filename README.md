# Efficient Chain-of-Thought

> **Can we decouple reasoning from human language to save tokens without sacrificing accuracy?**

This project shows that *compressed* reasoning traces—taught via LoRA fine-tuning—can match natural-language chain-of-thought while using **up to 80 % fewer tokens**.

| Dataset | Approach | Accuracy / F1 | Token Reduction |
|---------|----------|---------------|-----------------|
| GSM8K   | State Machine | 86.7 % (vs 90.9 %) | ~50 % |
| DROP    | State Machine | 69.4 F1 (vs 71.2) | ~82 % |

We experiment with two compression schemes—**Cipher** and **State Machine**—and further optimize with reinforcement learning (GRPO) for brevity + correctness. Built on [Tinker](https://thinkingmachines.ai/tinker/).

📄 **Full write-up:** [report.md](report.md)

---

<details>
<summary><strong>📊 Main Results Table</strong></summary>

| Configuration | GSM8K Accuracy (%) | Reasoning Tokens | DROP F1 (%) | Reasoning Tokens |
|---------------|-------------------|------------------|-------------|------------------|
| **Zero-Shot** |
| Qwen3-8B (no reasoning) | 13.8 ± 1.0 | — | 44.8 ± 1.4 | — |
| Qwen3-8B (standard CoT) | 61.1 ± 1.4 | 451.7 | 63.2 ± 1.3 | 507.9 |
| **SFT Baselines** |
| No reasoning | 69.7 ± 1.3 | — | 62.8 ± 1.3 | — |
| Standard CoT | **90.9 ± 0.8** | 124.1 | **71.2 ± 1.3** | 320.3 |
| **SFT Compressed** |
| Cipher | 83.2 ± 1.0 | 57.9 | 69.2 ± 1.3 | 112.7 |
| State Machine | 86.7 ± 0.9 | 61.7 | 69.4 ± 1.3 | 58.2 |
| **SFT + RL** |
| Standard CoT | 90.2 ± 0.8 | 57.2 | **72.8 ± 1.2** | 54.6 |
| Cipher | 84.1 ± 1.0 | **44.3** | 72.1 ± 1.2 | **32.4** |
| State Machine | 87.1 ± 0.9 | 57.1 | 71.2 ± 1.2 | 45.7 |

*All models: Qwen3-8B, LoRA rank 32. SFT LR 5e-4; RL LR 2e-4 (standard) / 1e-4 (compressed).*

</details>

<details>
<summary><strong>🔍 Example: GSM8K Math Problem</strong></summary>

**Question:** Zaid spends 1/4 of his salary on rent, 1/3 on fuel, donates half the remainder to charity, gives \$200 to his daughter and \$700 to his wife. Salary = \$6000. How much is left?

**Ground Truth:** 350

| Human-Readable CoT | State Machine | Cipher |
|--------------------|---------------|--------|
| *"When Zaid spends 1/4 of his salary…"* (401 tokens) | `S0[6000] → S1[⊖1/4=1500]✓ → … → S7[350]✓` (129 tokens, **−68 %**) | `α κ6000 → θ1/4 → … → ν=350` (109 tokens, **−73 %**) |

✅ All three produce the correct answer.

</details>

---

## Quick Start

### 1. Setup

```bash
# Clone & create venv
git clone https://github.com/zxsimon/efficient-CoT && cd efficient-CoT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tinker API key (or add to .env)
export TINKER_API_KEY=sk-...
```

### 2. (Optional) Generate Compressed Datasets

Spin up a local inference endpoint (e.g., LM Studio at `http://127.0.0.1:1234`), then:

```bash
python -m dataset.generate --dataset gsm8k --approach sm1 cipher1
```

Prompts live in `dataset/prompts.jsonl`; outputs land in `dataset/{name}_{approach}_{split}.jsonl`.

### 3. Train

**Supervised Fine-Tuning**

```bash
python -m train.sft --dataset gsm8k_sm1 --lr 5e-4 --rank 32
```

**Reinforcement Learning**

```bash
# Requires a prior SFT run e.g. logs/sft/gsm8k_sm1_32_0.0005. 
# Otherwise, leave --model flag empty to RL a base model
python -m train.rl --model gsm8k_sm1_32_0.0005 --lr 1e-4
```

Logs & checkpoints → `logs/{sft,rl}/...`

All other hyparameters can be provided as additional CLI flags or edited directly in `train/sft.py` and `train/rl.py`.

### 4. Evaluate & Plot

```bash
# Download weights first: tinker checkpoint download {state_path}
# Then save into checkpoints/ e.g. checkpoints/gsm8k_sm1_32_5e-4
python -m eval.eval --checkpoint gsm8k_sm1_32_5e-4 --batch_size 8

# Training curves
python -m results.plots --sft
python -m results.evals --prefix eval_gsm8k
```

---

## All in all…

<p align="center">
  <img src="https://media.tenor.com/images/cc242f1691c5783b2ecd23d7779c8a13/tenor.gif" alt="Why waste time say lot word when few word do trick" width="300"/>
</p>
