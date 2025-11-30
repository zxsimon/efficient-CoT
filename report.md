# **Efficient Chain-of-Thought: Decoupling Reasoning from Human Language**

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Background and Related Works](#2-background-and-related-works)  
3. [Methodology](#3-methodology)  
4. [Experimental Results](#4-experimental-results)  
5. [Limitations and Future Work](#5-limitations-and-future-work)  
6. [Conclusion](#6-conclusion)  
7. [References](#7-references)

# **1. Introduction**

Chain-of-Thought (CoT) reasoning dramatically improves language model performance on complex tasks by making intermediate reasoning explicit. However, standard CoT uses verbose natural language (100-500 tokens per problem), creating a fundamental tension: optimizing simultaneously for task accuracy *and* human interpretability may constrain both performance and efficiency.

**Research Question.** Can we compress reasoning substantially while maintaining accuracy—either through symbolic representations or by removing the human-readability constraint from natural language itself?

**Approach.** We train Qwen3-8B via LoRA on two compression strategies—*cipher* (symbolic token mappings: α, κ, θ) and *state machine* (structured transitions: S₀ → S₁ → S₂)—across GSM8K (math) and DROP (reading comprehension). Training combines supervised fine-tuning to teach compressed grammars, then reinforcement learning (GRPO) to optimize for both accuracy and token efficiency.

**Key Findings.**

1. **Compressed symbolic reasoning works:** State machine achieves 95-97% of standard CoT accuracy while reducing tokens by 50-82%, demonstrating that mathematical and extractive reasoning can accommodate substantial compression.

2. **Verbosity is the bottleneck:** When RL optimizes natural language CoT for efficiency, it achieves up to 83% token reduction with no accuracy loss, producing concise but still-readable reasoning traces. This reveals that redundant explanations and verbose phrasings—not the use of natural language—were the primary inefficiency.

3. **Stability-efficiency tradeoff:** State machine trains reliably; cipher achieves lower token counts but exhibits instability (loops, length spikes). RL-compressed natural language offers a middle ground: competitive efficiency with greater robustness and interpretability.

These results demonstrate that reasoning efficiency gains come primarily from removing explanatory verbosity, whether through symbolic compression or RL optimization of natural language.

# **2. Background and Related Works**

## **2.1 Theoretical Framework and Motivation**

The optimization of Large Language Models (LLMs) centers on modeling the conditional probability of generating a correct answer $y$ (e.g., "42") given an input question $x$ (e.g., "What is 6 × 7?"), parameterized by model weights $\theta$. Standard training minimizes the negative log-likelihood:

$$\mathcal{L} = \mathbb{E}_{(x,y) \sim \mathcal{D}}[-\log P_{\theta}(y | x)]$$

where $\mathcal{D}$ is the training distribution. We can decompose this conditional probability by introducing intermediate reasoning steps $z$:

$$P_{\theta}(y|x) = \sum_{z \in \mathcal{Z}} P_{\theta}(z|x)P_{\theta}(y|x,z) $$

where:
- $x$ represents the input question or problem
- $y$ represents the final answer
- $z$ represents an intermediate reasoning chain
- $\mathcal{Z}$ is the space of all possible reasoning chains
- $\theta$ denotes the model parameters

Standard Chain-of-Thought approaches make $z$ explicit as human-readable natural language sequences $\tilde{z}$ (e.g., "First, multiply 6 and 7 to get 42"). However, optimizing for both task accuracy and human interpretability may create competing objectives. If the training procedure includes an implicit constraint to maximize $P(\tilde{z}|x)$ for readability, this could potentially bound the achievable task performance $P(y|x,\tilde{z})$.

We hypothesize that discrete but semantically-compressed representations $z'$ (e.g., using symbolic tokens like "α ⊗ 6,7 → 42") can achieve comparable task performance $P(y|x,z') \approx P(y|x,\tilde{z})$ while requiring significantly fewer tokens. This approach preserves the discrete scratchpad benefits of explicit CoT while removing the human-readability constraint.

## **2.2 Related Works**

### Explicit Chain-of-Thought
Recent advances in reasoning-capable language models have been driven largely by explicit Chain-of-Thought approaches. Most notably, DeepSeek-R1 [^1] employs reasoning-oriented reinforcement learning with cold-start data from supervised fine-tuning, achieving remarkable performance on complex reasoning tasks. However, DeepSeek-R1's training explicitly optimizes for human-readable reasoning traces as part of its design philosophy to ensure interpretability and alignment with human reasoning patterns.
This design choice, while valuable for model interpretability and safety, raises an important question: does enforcing human readability impose a constraint on the model's optimization objective? If the reinforcement learning procedure includes an objective to maximize the probability of interpretable reasoning traces, the achievable task performance may be bounded—what we term a "human-constraint upper bound." In other words, requiring reasoning to be legible to humans may limit the model's ability to discover more efficient or effective reasoning pathways.
Beyond interpretability constraints, the auto-regressive token generation process itself may introduce computational inefficiencies. At each generation step, the model must project its high-dimensional hidden state into vocabulary space, select a token, and re-embed it—creating an information bottleneck. This could explain phenomena such as DeepSeek-R1-Zero's tendency to increase reasoning length during training: the model may be finding circuitous but comprehensible token sequences to access additional computational layers, rather than discovering truly optimal reasoning paths. Recent work has begun questioning whether semantic coherence is necessary for effective reasoning. Studies on "dot-by-dot" reasoning [^2] and pause tokens [^3] demonstrate that the volume of computation matters more than semantic content, suggesting that human-interpretable language may not be essential.

### Implicit and Latent Reasoning Approaches
An alternative paradigm has emerged through models that reason in continuous latent space rather than discrete tokens. COCONUT [^4] introduces latent "thought tokens" that exist only in the model's hidden states, while CODI [^5] employs diffusion processes in latent space for reasoning. These approaches seek representations that potentially surpass human-readable CoT in both efficiency and performance.
One compelling advantage of latent reasoning is the possibility of parallel exploration similar to tree search algorithms. By reasoning in continuous space, models may avoid the "tunnel vision" induced by auto-regressive generation, where each token commits the model to a specific reasoning branch. However, these approaches sacrifice the interpretability and discrete scratchpad benefits that make explicit CoT valuable for debugging, verification, and human oversight.

### Our Approach: Compressed Discrete Reasoning
This work occupies a middle ground between explicit human-readable and fully implicit reasoning. We train models via LoRA to use discrete tokens in semantically compressed ways—creating a discrete scratchpad that preserves the structural benefits of explicit CoT while removing the human-readability constraint. Rather than reasoning in continuous latent space or verbose natural language, our models learn compressed symbolic representations. This approach aims to achieve comparable performance to human-readable CoT while significantly reducing token usage and potentially discovering more efficient reasoning pathways unconstrained by natural language structure.

# **3. Methodology**

Our approach consists of two training phases: (1) supervised fine-tuning (SFT) to teach the model compressed reasoning grammars, and (2) reinforcement learning (RL) to optimize for both accuracy and token efficiency. We use Low-Rank Adaptation (LoRA) throughout to enable efficient parameter updates while preserving the base model's capabilities. All experiments use Qwen3-8B [^6] as the base model, selected for its strong reasoning capabilities and computational feasibility.

## **3.1 Dataset Selection and Generation**

#### Figure 1: Overview of Synthetic Data Generation Process

<p align="center">
<img src="misc/data.png" width="50%">
</p>

**Mathematical Reasoning: GSM8K.** We began with the Grade School Math 8K (GSM8K) mathematics dataset [^7], which provides 7,473 training examples and 1,319 test examples of grade-school math problems. GSM8K is particularly well-suited for studying compressed reasoning because (1) mathematical reasoning has clear correctness criteria, (2) problems require multi-step logical chains, and (3) the dataset provides human-written chain-of-thought solutions that we can compress.

**General Reasoning: DROP.** While GSM8K validates our approach on mathematical reasoning, we sought to demonstrate generalizability to natural language reasoning tasks. This led us to search for a dataset that required reasoning over English text rather than pure arithmetic. However, finding a suitable second dataset proved challenging: we needed problems that (1) genuinely required reasoning (not just extraction), (2) had minimal external context requirements, and (3) provided enough signal to distinguish compressed from readable reasoning. 

After comparing alternatives such as MMLU, CommonsenseQA, StrategyQA, ProntoQA, and HellaSwag, we ultimately selected DROP (Discrete Reasoning Over the content of Paragraphs) [^8] as our second dataset, as it combines reading comprehension with numerical reasoning while requiring minimal external knowledge retrieval. 

During preliminary experiments, we observed an interesting characteristic of DROP: our SFT-trained base model achieved 63% F1 score without any reasoning traces, compared to 71% F1 with reasoning — a difference of only 8 percentage points. This contrasts with GSM8K, where no-reasoning achieves 70% while reasoning achieves 91% - a 21 point gap. The smaller reasoning benefit in DROP suggests that many questions can be answered through pattern matching or direct extraction, which has important implications for our analysis: improvements in F1 score may be noisier and harder to attribute definitively to reasoning efficiency gains. We include DROP results to explore how compressed reasoning performs across different reasoning requirements, while acknowledging that the signal may be weaker than in GSM8K.

**Synthetic Data Generation.** For both datasets, we generated compressed reasoning traces using Qwen3-30B-A3B-2507, a larger and more capable model in the Qwen family. We experimented with two compression strategies, illustrated through the following GSM8K example:

**Example Question:**
> Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

**Original Reasoning (Standard CoT, Human-Readable):**
> Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.

Our compression strategies transform this reasoning as follows:

**1. Cipher:** Semantic tokens are mapped to Greek letters and mathematical symbols, creating a symbolic language that preserves logical operations while eliminating natural language redundancy.

*Compressed form:* `α: κ48 → α θ(48÷2)=24 → α κ48+24=72`

*Symbol mapping:* α (compute), κ (quantity), θ (operation), → (step transition)

**2. State Machine:** Reasoning is represented as transitions between abstract states, where each state captures intermediate computational results.

*Compressed form:* `S0[48] → S1[÷2=24] → S2[48⊕24=72]✓`

*Notation:* $S_i$ (state i), [operation], → (transition), ✓ (final answer)

Both strategies preserve the logical structure and computational steps of the original reasoning while achieving significant token reduction (20-25% for the above GSM8k example). We also generated a no-reasoning baseline where the model outputs empty thinking blocks `<think>\n\n</think>`, allowing us to isolate the value of explicit reasoning versus direct answer generation. Importantly, while we ensured internal consistency within each problem's reasoning trace, we did not enforce strict symbol usage consistency *across* problems. This design choice allows the model to potentially discover its own optimal reasoning grammar during training, rather than being constrained to the specific symbolic conventions used in the synthetic data generation.

**Controlling for Selection Bias in DROP Dataset Curation.** The DROP dataset presented unique curation challenges due to its diverse question types and answer formats. To ensure high-quality reasoning traces, we applied an F1 score threshold (F1 > 0.8) when generating synthetic reasoning with Qwen3-30B, filtering for cases where the model demonstrated strong comprehension of the passage and question. However, this filtering process introduces a potential selection bias: the curated training set may not be representative of the full DROP distribution. To address this concern, we took two precautionary measures:

1. **Distribution Verification:** We verified that the textual v.s. numerical answer distribution in our F1-filtered subset remained consistent with the overall DROP dataset, ensuring we did not systematically favor certain question types.

2. **Unbiased Evaluation:** While we train on the F1-filtered dataset and compute test negative log-likelihood (NLL) on this filtered set, we evaluate **accuracy on a held-out, non-filtered validation set**. This ensures that our reported performance metrics reflect the model's capability on the full DROP distribution, not just the easy-to-answer subset used for training. This separation between training data curation and evaluation methodology prevents our F1 filtering from artificially inflating accuracy results.

## **3.2 Model Architecture**

**Base Model:** We use Qwen3-8B, an 8-billion parameter decoder-only transformer trained on a diverse corpus of text and code. The model employs grouped-query attention and uses the Qwen tokenizer with a vocabulary of 151,646 tokens.

**LoRA Configuration:** We apply Low-Rank Adaptation to all attention layers (query, key, value, and output projections). We experiment with LoRA ranks of [16, 32, 64] during ablation studies, ultimately selecting rank 32 as it provided a good balance between expressiveness and training stability.

**Training Infrastructure:** All training was conducted on Thinking Machine's Tinker API. Evaluation was done on a single NVIDIA H100 PCIe GPU.

## **3.3 Training Pipeline**

### Phase 1: Supervised Fine-Tuning
In the first phase, we teach the model the "grammar" of compressed reasoning through supervised learning on our synthetic datasets. The model learns to map questions to compressed reasoning traces and final answers:
```
Question → <think> compressed_reasoning </think> → Answer
```

We train for 5 epochs on reasoning datasets with a batch size of 32 and a constant learning rate of 5e-4. We monitor both training and test negative log-likelihood (NLL), selecting the checkpoint with lowest test NLL for RL initialization.

### Phase 2: Reinforcement Learning
In the second phase, we apply Group Relative Policy Optimization (GRPO) to optimize for both task accuracy and token efficiency. We use a batch size of 32 with group size of 16, performing a single epoch over each training set (~8,000 examples for both GSM8K and DROP, corresponding to ~250 training steps). Learning rates are set to 2e-4 for human-readable reasoning and 1e-4 for compressed reasoning, reflecting the latter's sensitivity to learning rate as observed during supervised fine-tuning.

Our reward function balances answer correctness and reasoning efficiency:

$$\text{reward} = \begin{cases}
s - \alpha \cdot \min\left(\frac{\ell}{\ell_{\max}}, 1\right) & \text{if } s \geq \tau \\
0 & \text{otherwise}
\end{cases}$$

where $s$ is the correctness score (binary for GSM8K, F1 for DROP), $\alpha$ is the maximum penalty (set to 1.0), $\ell$ is the reasoning token count, $\ell_{\max}$ is the maximum reasoning length threshold, and $\tau$ is the correctness threshold (1.0 for GSM8K, 0.8 F1 for DROP). This formulation penalizes only correct answers for verbosity—incorrect answers receive zero reward regardless of length.

The critical hyperparameter is $\ell_{\max}$, which determines when a correct answer would receive zero reward. For GSM8K we set $\ell_{\max} = 300$ tokens, meaning a correct answer using the full 300 reasoning tokens would achieve a reward of 0. For DROP, we set $\ell_{\max} = 700$ tokens.

**Alternative Reward Mode.** We also implemented a "penalize all" mode where incorrect answers incur length penalties, but due to computational constraints, we focus our evaluation on the "penalize correct only" formulation, which provides more stable training dynamics.

## **3.4 Baseline Configurations**

For each dataset, we evaluate our compressed reasoning approaches against several baselines to isolate the contributions of fine-tuning, reasoning, and compression:

**Zero-Shot Baselines (Pre-Training Only):**

1. **Zero-Shot Base Model (No Reasoning):** Qwen3-8B with no task-specific fine-tuning, where the prompt includes an empty thinking block `<think>\n\n</think>`, forcing direct answer generation without reasoning. This establishes the model's inherent capability when reasoning is explicitly suppressed.

2. **Zero-Shot Base Model (Free Reasoning):** Qwen3-8B allowed to freely generate its own thinking block content. This reveals whether the base model spontaneously produces useful reasoning patterns without any fine-tuning on reasoning traces.

**Supervised Fine-Tuning (SFT) Baselines:**

3. **SFT No-Reasoning:** Model fine-tuned to consistently output empty thinking blocks `<think>\n\n</think>` before answers. Isolates the effect of task-specific fine-tuning without explicit reasoning.

4. **SFT Human-Readable CoT:** Model fine-tuned on natural language reasoning traces, representing the "human-constraint" baseline. This is our primary comparison target—we aim to match this performance with compressed representations.

All SFT configurations use identical training procedures and hyperparameters for fair comparison. The zero-shot baselines allow us to measure the value of fine-tuning, while comparisons between SFT variants isolate the effects of reasoning content and compression.


# **4. Results and Discussion**

### Overall Results

#### Table 1: GSM8K and DROP Test Set Performance Comparison for SFT.

| Configuration | GSM8K Accuracy (%) | GSM8k Reasoning Length | DROP F1 Score (%) | DROP Reasoning Length |
|---------------|--------------|------------------|--------------|------------------|
| **Zero-Shot Baselines** | | |
| Qwen3-8B (no reasoning) | 13.8 (±1.0) | — | 44.8 (±1.4) | -
| Qwen3-8B (standard CoT) | 61.1 (±1.4) | 451.7 (±9.7) | 63.2 (±1.3) | 507.9 (±8.9)
| **Supervised Fine-Tuning: Baseline Models** | | |
| SFT no reasoning | 69.7 (±1.3) | - | 62.8 (±1.3) | -
| SFT with standard CoT | **90.9 (±0.8)** | 124.1 (±1.6) | **71.2 (±1.3)** | 320.3 (±6.2)
| **Supervised Fine-Tuning: Compressed Reasoning Models** | | |
| SFT with Cipher CoT | 83.2 (±1.0) | 57.9 (±1.5) | 69.2 (±1.3) | 112.7 (±6.5)
| SFT with State Machine CoT | 86.7 (±0.9) | 61.7 (±1.5) | 69.4 (±1.3) | 58.2 (±2.2)
| **Supervised Fine-Tuning + Reinforcement Learning Models** | | |
| SFT + RL with standard CoT | 90.2 (±0.8) | 57.2 (±1.5) | **72.8** (±1.2) | 54.6 (±1.9)
| SFT + RL with Cipher CoT | 84.1 (±1.0) | **44.3** (±0.9) | 72.1 (±1.2) | **32.4** (±0.9)
| SFT + RL with State Machine CoT | 87.1 (±0.9) | 57.1 (±1.7) | 71.2 (±1.2) | 45.7 (±0.7)

*All models use Qwen3-8B with Rank 32 LoRA. All SFT models used LR 5e-4. RL with standard CoT used LR 2e-4, while RL with Cipher/Statement used LR 1e-4. Standard errors shown in parentheses. Reasoning length measured in tokens.*

## **4.1 Supervised Fine-Tuning**

### Discussion of Results

Table 1 reveals dataset-dependent differences in reasoning requirements and compression tolerance. On GSM8K, compressed reasoning achieves 95% of standard CoT performance (86.7% vs 90.9%) with 50% token reduction, demonstrating that mathematical reasoning can accommodate substantial compression while maintaining strong accuracy. State machine outperforms cipher by 3.5 percentage points, suggesting that structured state transitions provide clearer optimization signals than flexible symbolic mappings.

On DROP, both compression strategies achieve 97% of standard CoT performance (69.4% vs 71.2%) with even greater efficiency gains—82% token reduction for state machine (58.2 vs 320.3 tokens). However, the narrow performance gap between compressed and standard reasoning (1.8 points) compared to GSM8K (4.2 points), combined with the small absolute difference between reasoning and no-reasoning baselines (69.4% vs 62.8%), reinforces our earlier observation that DROP's reasoning requirements are limited. The finding that fine-tuning alone (no reasoning) nearly matches zero-shot with full reasoning (62.8% vs 63.2%) further suggests that task-specific adaptation captures much of DROP's performance gains, with explicit reasoning providing only marginal additional benefit.

These results establish that compressed reasoning is both feasible and efficient, but its relative advantage depends critically on the depth of reasoning required by the task—substantial for mathematical problems like GSM8K, more modest for pattern-matching tasks like DROP.

---
### Raw Model Outputs on Test Set

Here we present a few example model outputs from test set (click on each to expand). 

<details>
<summary><strong>Example 1: Mathematical Reasoning (GSM8k) - All Correct ✅</strong></summary>

---

**Question:**

> Zaid spends 1/4 of his salary on rent, 1/3 on car fuel and donates half of the remaining amount to his favorite charity. He gives his daughter 200$ to use for her weekly expenses and 700$ to his wife to budget for groceries and other household goods. If Zaid earns 6000$ per month, how much money will he still have after all these expenses and donations? Provide ONLY a final numerical answer, with no explanation, no units, no punctuation, and no other text.

**Ground Truth:** 350

---

| Human-Readable CoT | State Machine | Cipher |
|--------------------|---------------|--------|
| When Zaid spend 1/4 of his salary on rent, he parts with 1/4*$6000 = $1500. Car fuel will cost him 1/3*$6000 = $2000. Total expenses amount to $2000+$1500 = $3500. Car fuel and rent expenses combined amount to $3500. When he subtracts these expenses from his salary, he remains with $6000-$3500 = $2500. He donates half of the remaining amount, which totals 1/2*$2500 = $1250. The total expenses and donations amount to $1250+$3500 = $4750. When Zaid subtracts his expenses and donations from his salary, he remains with $6000-$4750 = $1250. He also gives his daughter $200, reducing the amount to $1250-$200 = $1050. Finally, after giving his wife $700, Zaid remains with $1050-$700 = $350 | S0[6000] → S1[⊖1/4=1500]✓ → S2[⊖1/3=2000]✓ → S3[1500⊕2000=3500]✓ → S4[6000⊖3500=2500]✓ → S5[⊘2=1250]✓ → S6[1250⊖200=1050]✓ → S7[1050⊖700=350]✓ | α κ6000 → θ1/4 → α κ1500 → θ1/3 → α κ2000 → ν=1500+2000=3500 → μ=6000-3500=2500 → φ1/2 → α κ1250 → δ: 200+700=900 → ν=1250-900=350 |
| ✅ **Prediction: 350** | ✅ **Prediction: 350** | ✅ **Prediction: 350** |
| 📏 401 tokens | 📏 129 tokens (⚡ 68% reduction) | 📏 109 tokens (⚡ 73% reduction) |

> 📊 **Comments:** All methods produce correct answer. Cipher compresses reasoning slightly more efficiently.

---

</details>

<details>
<summary><strong>Example 2: Historical Reading Comprehension (DROP) - All Correct ✅</strong></summary>

---

**Question:**

> The long border between Burma and China had long been vaguely defined. The Ming dynasty first conquered Yunnan borderlands between 1380 and 1388, and stamped out local resistance by the mid-1440s. The Burmese control of the Shan States came in 1557 when King Bayinnaung of the Toungoo dynasty conquered all of the Shan States. The border was never demarcated in the modern sense, with local Shan sawbwas at the border regions paying tribute to both sides. The situation turned to China's favor in the 1730s when the Qing decided to impose a tighter control of Yunnan's border regions while the Burmese authority largely dissipated with the rapid decline of the Toungoo dynasty. What title of the Toungoo dynasty did Bayinnaung have? Provide ONLY a concise final answer, with no explanation, no punctuation, and no other text.

**Ground Truth:** King

---

| Human-Readable CoT | State Machine | Cipher |
|--------------------|---------------|--------|
| The question asks for the title of the Toungoo dynasty that Bayinnaung held. We are given historical context: Bayinnaung is described as "King Bayinnaung of the Toungoo dynasty," and the text states that he conquered all of the Shan States in 1557. The key point is that the title "King" is explicitly used in the text — "King Bayinnaung of the Toungoo dynasty." This is a direct and unambiguous reference to his title. In Burmese (Myanmar) history, the Toungoo dynasty was the ruling dynasty that ruled Burma from the 16th to the 17th century. Bayinnaung was the most prominent ruler of the Toungoo dynasty, known for unifying Burma and the Shan States. His title as "King" is consistent with the historical records of the time. The text does not use any alternative title such as "Emperor" or "Lord of the Land," and the phrase "King Bayinnaung of the Toungoo dynasty" is directly stated. Therefore, the title in question is simply "King." | S0[Bayinnaung → Toungoo dynasty] → S1[text: "King Bayinnaung"] → S2[✓ title = King] | α: Bayinnaung → β: Toungoo dynasty → γ: "King Bayinnaung of the Toungoo dynasty" → δ: extract title → ε: King |
| ✅ **Prediction: King** | ✅ **Prediction: King** | ✅ **Prediction: King** |
| 📏 289 tokens | 📏 37 tokens (⚡ 87% reduction) | 📏 71 tokens (⚡ 75% reduction) |

> 📊 **Comments:** All three methods succeed on this straightforward extraction task. Human-readable CoT seems to be referencing some historical context, which might or might not have aided the model in its answering process.

---

</details>

<details>
<summary><strong>Example 3: Basic Statistical Processing (DROP) - State Machine Solo Win ⚠️</strong></summary>

---

**Question:**

> As of the census of 2000, there were 19,234 people, 7,473 households, and 4,954 families residing in the village. The population density was 10,337.3 people per square mile (3,992.6/km²). There were 7,650 housing units at an average density of 4,111.5/sq mi (1,588.0/km²). The racial makeup of the village was 82.39% White (U.S. Census), 1.03% African American (U.S. Census), 0.29% Native American (U.S. Census), 4.52% Asian (U.S. Census), 0.04% Pacific Islander (U.S. Census), 3.93% from Race (United States Census), and 3.79% from two or more races. 13.03% of the population is Hispanic (U.S. Census) or Latino (U.S. Census) of any race. Which racial group was the second smallest? Provide ONLY a concise final answer, with no explanation, no punctuation, and no other text.

**Ground Truth:** Native American

---

| Human-Readable CoT | State Machine | Cipher |
|--------------------|---------------|--------|
| We are given the racial makeup of the village as follows (from the 2000 census): 82.39% White, 1.03% African American, 0.29% Native American, 4.52% Asian, 0.04% Pacific Islander, 3.93% from Race (United States Census) — this likely refers to "Other" or unspecified race... the phrasing "3.93% from Race" is ambiguous and likely a misstatement. In standard U.S. Census terminology, "from Race" is not a standard category... Let's check if the percentages add up: 82.39 + 1.03 = 83.42 + 0.29 = 83.71 + 4.52 = 88.23 + 0.04 = 88.27 + 3.93 = 92.20 + 3.79 = 95.99%. So total is about 96%, which suggests some rounding or mislabeling. But the key is: which group is the second smallest? From the list: Pacific Islander: 0.04% → smallest, Native American: 0.29% → next smallest... 5. Native American (0 *(truncated at max tokens)* | S0[White:82.39%, 2+ races:3.79%, Asian:4.52%, Native Am:0.29%, Pacific Islander:0.04%, African Am:1.03%, Hispanic:13.03%] → S1[0.04% < 0.29% < 1.03% < 3.79% < 4.52% < 82.39%] → S2[2nd smallest:0.29% → Native American] | α: κ82.39% (White), λ1.03% (African American), μ0.29% (Native American), ν4.52% (Asian), ξ0.04% (Pacific Islander), π3.93% (Race), ψ3.79% (Two or more races) → Sort: ξ0.04%, μ0.29%, π3.79%, ψ3.79%, ν4.52%, λ1.03%, α82.39% → Second smallest: π3.79% → Answer: Race |
| ❌ **Prediction: N/A** | ✅ **Prediction: Native American** | ❌ **Prediction: Race** |
| 📏 1023 tokens (Max) | 📏 146 tokens | 📏 139 tokens |

> ⚠️ **Comments:** Admittedly, this is a strange question from the DROP dataset which seems to have a typo, as pointed out by the human-reasoning CoT. This red herring, intentional or not, distracted the human-reasoning CoT from getting to an answer within 1024 thinking tokens, although it seemingly almost gets there in the end. Both State Machine and Cipher used sorting pretty efficiently, with the former getting the right answer.

---

</details>

<details>
<summary><strong>Example 4: Self-Correcting State Machine (GSM8k) - All Correct ✅</strong></summary>

---

**Question:**

> Farmer Brown has 20 animals on his farm, all either chickens or cows. They have a total of 70 legs, all together. How many of the animals are chickens? Provide ONLY a final numerical answer, with no explanation, no units, no punctuation, and no other text.

**Ground Truth:** 5

---

| Human-Readable CoT | State Machine | Cipher |
|--------------------|---------------|--------|
| Let C be the number of chickens. There are 20-C cows. The cows have 4*(20-C) legs. The chickens have 2*C legs. The total is 2*C+4*(20-C)=70 legs. 2C+80-4C=70. 2C=10. C=5 | S0[total:20, legs:70] → S1[20−x −x=70] → S2[20−2x=70] → S3[−2x=50] → S4[x=−25] → S5[invalid] ❌ Wait — error in logic. Let's correct: Let x be the number of cows. Then 20−x is the number of chickens. Total legs: 2x + 2(20−x) = 70 → 40 = 70 → contradiction. Wait — that can't be. Let's try again. Let x be the number of cows. Each cow has 4 legs, each chicken has 2. So: 4x + 2(20−x) = 70 → 2x + 40 = 70 → 2x = 30 → x = 15. Then chickens: 20−15 = 5. Verify: 15 cows ×4 = 60, 5 chickens ×2 = 10 → 60+10=70. Correct. So answer is 5. | α κ20 → β λ70 → γ κ20-x → δ κ2x + 4(20-x) =70 → 2x +80-4x=70 → -2x=-10 → x=5 → α κ5 |
| ✅ **Prediction: 5** | ✅ **Prediction: 5** | ✅ **Prediction: 5** |
| 📏 80 tokens | 📏 399 tokens (⚠️ 399% increase) | 📏 60 tokens (⚡ 25% reduction) |

> 📊 **Comments:** A bit of a not-so-trivial algebra question this time. State Machine exhibits a fascinating behavior: it made an initial error when reasoning in its compressed syntax, but detected it mid-reasoning, and started backtracking and reasoning out in human-readable reasoning. Its use of human-readable CoT as a fallback is quite curious - this could suggest that the model can handle multiple reasoning grammars, and learn to use the most appropriate or efficient one depending on the task at hand.

---

</details>

<details>
<summary><strong>Example 5: Cipher Repetitive Loop Failure (DROP) - Human-Readable Solo Win ❌</strong></summary>

---

**Question:**

> Hoping to rebound from their road loss to the Patriots, the Rams went home for a Week 9 NFC West duel with the Arizona Cardinals. In the first quarter, St. Louis struck first as QB Marc Bulger completed an 80-yard TD pass to WR Derek Stanley. In the second quarter, the Cardinals responded with a vengeance as safety Antrel Rolle returned an interception 40 yards for a touchdown, kicker Neil Rackers got a 36-yard field goal, RB Tim Hightower got a 30-yard TD run, and former Rams QB Kurt Warner completed a 56-yard TD pass to WR Jerheme Urban. In the third quarter, Arizona increased its lead as Warner completed a 7-yard TD pass to WR Anquan Boldin. In the fourth quarter, the Rams tried to come back as Bulger completed a 3-yard TD pass to WR Torry Holt (with a failed 2-point conversion). However, the Cardinals flew away as Rackers nailed a 30-yard field goal. During the game, the Rams inducted former Head Coach Dick Vermeil (who helped the franchise win Super Bowl XXXIV) onto the Rams Ring of Honor. Which team got the first turnover? Provide ONLY a concise final answer, with no explanation, no punctuation, and no other text.

**Ground Truth:** Cardinals

---

| Human-Readable CoT | State Machine | Cipher |
|--------------------|---------------|--------|
| To determine which team got the first turnover, we need to examine the sequence of events in the game as described. The game begins with the Rams scoring first in the first quarter: QB Marc Bulger completes an 80-yard TD pass to WR Derek Stanley. This is a touchdown, not a turnover. In the second quarter, the Cardinals respond with a vengeance. The text states: "safety Antrel Rolle returned an interception 40 yards for a touchdown." This indicates that the Cardinals recovered an interception — a turnover — and scored on it. Since the Rams scored first (a touchdown), and the Cardinals responded with an interception return (a turnover), the Cardinals got the first turnover. | S0[Q: first turnover?] → S1[1Q: Bulger→80yd TD (no TO)] → S2[2Q: Rolle→40yd INT TD (TO)]✓ → S3[Cardinals got first TO] ⟹ S4[✓Arizona] | α: λ80yd TD pass → β: κ36yd FG, μ30yd TD, ν56yd TD → ξ: λ7yd TD → α: λ3yd TD → κ30yd FG → α κ80yd TD → β κ36yd FG... *(repeating loop ~20 times until max tokens)* |
| ✅ **Prediction: cardinals** | ⚠️ **Prediction: Arizona** *(correct team, wrong format)* | ❌ **Prediction: rams** 🔄 *(stuck in infinite loop)* |
| 📏 184 tokens | 📏 63 tokens | 📏 1000 tokens (Max) |

> ❌ **Comments:** Cipher suffered a catastrophic failure, getting stuck in a repetitive symbolic loop that consumed all 1000 tokens without ever addressing the actual question. This exemplifies how overly compressed symbolic reasoning can degenerate into meaningless gibberish without a halting condition when the system loses semantic grounding.

---

</details>


---

### Training Dynamics

We analyze training curves across all three reasoning formats on DROP to understand compression learnability and optimization characteristics. Training curves are generally similar for GSM8k, with greater training stability likely due to task simplicity and consistency. We plot raw NLL values without smoothing in Figures 2–4.

#### Figure 2: Training dynamics for human-readable Chain-of-Thought on DROP (rank 32, LR 5e-4). Test Sample Size n = 20.

<p align="center">
<img src="results/plots/sft_drop_32_0.0005.png" width="50%">
</p>

Figure 2 shows that human-readable reasoning exhibits stable optimization dynamics. Train and test NLL decrease smoothly from ~0.6 to ~0.1, demonstrating successful learning of natural language reasoning patterns specific to this task. Reasoning length remains stable at ~350-450 tokens throughout training. Test accuracy fluctuates between 65-80% but shows no systematic improvement beyond step 200.

#### Figure 3: Training dynamics for state machine compressed reasoning on DROP (rank 32, LR 5e-4). Test Sample Size n = 20.

<p align="center">
<img src="results/plots/sft_drop_sm1_32_0.0005.png" width="50%">
</p>

Like the standard human-readable reasoning, Figure 3 shows that state machine compression demonstrates highly stable training dynamics. NLL decreases steadily while reasoning length remains remarkably constant at ~50 tokens throughout all 1,200 training steps. This stability suggests that the state machine's structured transitions (S₀ → S₁ → S₂) provide clear gradient signals that the model can optimize consistently. Test accuracy plateaus around 70% within 200 steps, matching the pattern observed for human-readable CoT.


#### Figure 4: Training dynamics for cipher compressed reasoning on DROP (rank 32, LR 5e-4). Test Sample Size n = 20.

<p align="center">
<img src="results/plots/sft_drop_cipher1_32_0.0005.png" width="50%">
</p>

In contrast, Figure 4 reveals that cipher compression exhibits significant optimization instability. While NLL decreases similarly to other methods, the reasoning length (orange line) exhibits dramatic fluctuations, spiking unpredictably from ~50 tokens to 200+ tokens at various points during training (steps ~200, ~600, ~1100). These instabilities suggest difficulty with gradient control—the model struggles to maintain consistent symbolic token generation, occasionally producing verbose or degenerate reasoning traces.

#### Discussion

**1. Grammar Learnability Varies by Compression Strategy**

The stark contrast between state machine stability and cipher instability demonstrates that not all compression schemes are equally learnable. We hypothesize that the ease of learning a new reasoning grammar depends on two factors:

*Factor 1: Dataset Quality and Consistency*

Our synthetic reasoning dataset may contain inconsistent reasoning patterns across examples. State machine's rigid structure (S₀ → S₁ → S₂) enforces consistency, providing stable training signals. In contrast, cipher's more flexible symbol usage may amplify dataset inconsistencies, leading to noisy gradients and training instabilities. Furthermore, the quality of the human reasoning data used in generating synthetic reasoning is also important, since we are essentially conducting a form of knowledge distillation through our training process.

*Factor 2: Transferability from Pre-Trained Knowledge*

State machine notation (S₀, →, brackets) may better align with the base model's pre-existing knowledge of mathematical and logical notation learned during pre-training. Cipher's Greek letters (α, κ, θ) represent a more dramatic departure from natural language, potentially requiring more complete "re-learning" of token semantics, which manifests as optimization instability. Future compression strategies should consider gradient stability and alignment with pre-trained knowledge as important design criteria.

**2. Rapid Performance Plateau**

All three methods show accuracy and reasoning length saturation within 200 steps, despite continued NLL improvements through step 1,200. On the one hand, this pattern reinforces our initial finding that DROP's reasoning requirements are limited—once basic patterns are learned, additional optimization yields minimal accuracy gains. On the other hand, this observation raises a more fundamental question about the extent to which explicit CoT reasoning, compressed or otherwise, meaningfully contributes to model performance.

Returning to our theoretical framework from Section 2, we can formalize this question: does improving $P(z|x)$ (the model's ability to generate reasoning traces) significantly impact $P(y|x) = P(y|x,z)P(z|x)$ (the final answer accuracy)? A more general formulation is:

$$P(y|x) =\sum_{z\in Z} P(y|x,f(z))P(z|x)$$

where $f(\cdot)$ represents a functional transformation of the intermediate reasoning step $z$. If $P(y|x,f(z))$ is relatively insensitive to the exact form of $z \in Z$—that is, if many different reasoning traces lead to the same answer—then optimizing $P(z|x)$ through test NLL minimization may yield diminishing returns for accuracy.

More provocatively, recent work has suggested that $f(\cdot)$ may simply measure the *length* or *presence* of reasoning tokens rather than their semantic content. If this hypothesis holds, then the specific grammatical structure of CoT (compressed, symbolic, or natural language) becomes less critical than ensuring some form of intermediate computation exists. Our DROP results—where accuracy plateaus quickly across all reasoning formats despite continued optimization—provide empirical support for this perspective, at least for datasets with limited reasoning depth.



### Ablation Studies

**Performance.** We conducted ablation studies on the GSM8k dataset using the state machine reasoning configuration, focusing on the key LoRA hyperparameters of rank (16, 32, 64) and learning rate (1e-4, 5e-4, 1e-3). We first picked a base learning rate of 5e-4 as suggested by recent research on optimal LoRA learning rate as a function of model size [^10]. As for rank, we began our experiments with 32 with the assumption that learning a new reasoning syntax might require more ranks than the more common choices of 8 or 16. Table 2 below summarizes test set evaluation results.


#### Table 2: LoRA Rank and Learning Rate Ablation (State Machine, GSM8K)

| Rank | LR = 1e-4 | LR = 5e-4 | LR = 1e-3 |
|------|---------|---------|---------|
| **16** | 88.3% / 65.1 tok | **89.4% / 61.3 tok** | ✗ |
| **32** | 87.9% / 63.5 tok | 86.7% / 61.7 tok | ✗ |
| **64** | 87.3% / 66.3 tok | 87.5% / 63.2 tok | ✗ |

*Format: Accuracy (%) / Mean reasoning length (tokens). Standard errors: ±0.85-0.94% for accuracy, ±1.05-1.68 tokens for length. ✗ indicates training divergence.* 

The most immediate observation is that LR = 1e-3 consistently diverges, producing unstable updates and gradient explosions (visualized in the next section). Among the stable configurations, rank 16 performs surprisingly well, achieving both strong accuracy and shorter reasoning traces, suggesting that higher ranks do not provide meaningful additional capacity for this task. The recommended learning rate of 5e-4 is competitive across ranks, but it does not yield a significant improvement over 1e-4 in either accuracy or reasoning efficiency.

**Training Convergence.**

#### Figure 5: LoRA Rank and Learning Rate Ablation (State Machine, GSM8K), Test NLL Convergence (left) and Final Test NLL (right)

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="results/plots/gsm8k_sm1_ablation_test_nll_sweep.png" width="45%">
  <img src="results/plots/gsm8k_sm1_ablation_final_nll.png" width="45%">
</div>


Figure 5 shows several patterns in compressed grammar acquisition. The plot on the left shows how learning rate substantially affects training stability. In particular, LR 1e-3 causes divergence across all ranks, while LR 5e-4 and 1e-4 both converge reliably to similar final NLL (~0.05-0.06). The instability at higher learning rates suggests that learning novel symbolic representations may require more conservative optimization than standard fine-tuning. Comparing between the two stable learning rates, we see that LR 5e-4 achieves faster early convergence, reaching lower NLL within the first 100 examples compared to LR 1e-4. 

On the right, we see that rank appears to minimally impact final convergence when learning rate is stable, with all three ranks achieving similar final NLL (variance <0.01). Combined with rank 16's superior test accuracy (Table 2), this suggests lower-rank adaptations may offer efficiency benefits without sacrificing optimization quality, though more systematic investigation would be needed to confirm this relationship, such as on the DROP dataset.


## **4.2 Reinforcement Learning**


<details><summary><strong>Table 1 from earlier for reference</strong></summary>

#### Table 1: GSM8K and DROP Test Set Performance Comparison for SFT.

| Configuration | GSM8K Accuracy (%) | GSM8k Reasoning Length | DROP F1 Score (%) | DROP Reasoning Length |
|---------------|--------------|------------------|--------------|------------------|
| **Zero-Shot Baselines** | | |
| Qwen3-8B (no reasoning) | 13.8 (±1.0) | — | 44.8 (±1.4) | -
| Qwen3-8B (standard CoT) | 61.1 (±1.4) | 451.7 (±9.7) | 63.2 (±1.3) | 507.9 (±8.9)
| **Supervised Fine-Tuning: Baseline Models** | | |
| SFT no reasoning | 69.7 (±1.3) | - | 62.8 (±1.3) | -
| SFT with standard CoT | **90.9 (±0.8)** | 124.1 (±1.6) | **71.2 (±1.3)** | 320.3 (±6.2)
| **Supervised Fine-Tuning: Compressed Reasoning Models** | | |
| SFT with Cipher CoT | 83.2 (±1.0) | 57.9 (±1.5) | 69.2 (±1.3) | 112.7 (±6.5)
| SFT with State Machine CoT | 86.7 (±0.9) | 61.7 (±1.5) | 69.4 (±1.3) | 58.2 (±2.2)
| **Supervised Fine-Tuning + Reinforcement Learning Models** | | |
| SFT + RL with standard CoT | 90.2 (±0.8) | 57.2 (±1.5) | **72.8** (±1.2) | 54.6 (±1.9)
| SFT + RL with Cipher CoT | 84.1 (±1.0) | **44.3** (±0.9) | 72.1 (±1.2) | **32.4** (±0.9)
| SFT + RL with State Machine CoT | 87.1 (±0.9) | 57.1 (±1.7) | 71.2 (±1.2) | 45.7 (±0.7)

*All models use Qwen3-8B with Rank 32 LoRA. All SFT models used LR 5e-4. RL with standard CoT used LR 2e-4, while RL with Cipher/Statement used LR 1e-4. Standard errors shown in parentheses. Reasoning length measured in tokens.*
</details>

### Discussion of Results

Building on our SFT baselines, we apply GRPO reinforcement learning to optimize models for both task accuracy and token efficiency. Table 1 (bottom section) presents our main RL results. The most unexpected finding concerns standard human-readable Chain-of-Thought reasoning. On GSM8K, RL achieves a 54% token reduction (124.1 → 57.2 tokens) while maintaining accuracy within 0.7 percentage points of the SFT baseline. On DROP, the compression is even more dramatic: an 83% reduction (320.3 → 54.6 tokens) accompanied by a slight *improvement* in F1 score (+1.6 points). This suggests that Qwen3-30B's synthetic reasoning traces contain substantial redundancy—verbose explanations and repetitive phrasings that contribute little to task performance. RL successfully identifies and eliminates this redundancy without sacrificing logical coherence, and in DROP's case, the forced brevity may even enhance clarity by removing distracting verbosity.

In contrast, our pre-compressed representations exhibit more modest efficiency gains under RL. Cipher achieves the lowest absolute token usage across both datasets (44.3 tokens on GSM8K, 32.4 on DROP), though on GSM8K this extreme compression comes at the cost of reduced accuracy (84.1% versus 90.2% for standard CoT). State machine maintains stronger accuracy (87.1% GSM8K, 71.2% DROP) while achieving intermediate token counts. However, we acknowledge an important limitation: our consistent length penalty threshold of $\ell_{\max} = 300$ tokens for GSM8K and $\ell_{\max} = 700$ tokens for DROP may not have been sufficiently aggressive for already-compressed representations. Since cipher and state machine start from 60-token baselines, a 300-token maximum provides little pressure for further optimization, potentially leaving additional efficiency gains unrealized.

---
### Raw Model Outputs on Test Set

These examples showcase the RL-trained models producing concise reasoning traces while still arriving at correct answers.

<details>
<summary><strong>Example 6: Mathematical Reasoning (GSM8K) - All Correct ✅</strong></summary>

---

**Question:**

> Johnny took his allowance of $20 and added an extra $10 to it. He then invested this sum of money, which tripled in a year. How much money did he have after a year? Provide ONLY a final numerical answer, with no explanation, no units, no punctuation, and no other text.

**Ground Truth:** 90

---

| Human-Readable CoT | Cipher | State Machine |
|--------------------|--------|---------------|
| 20+10=30, 30*3=90 | α κ$20+$10=$30 → α θ3 → α κ30×3=$90 | S0[20+10=30] → ⊗3=90✓ |
| ✅ **Prediction: 90** | ✅ **Prediction: 90** | ✅ **Prediction: 90** |
| 📏 17 tokens | 📏 27 tokens | 📏 21 tokens |

> 📊 **Comments:** Standard CoT stripped everything but the core equations, yielding the most efficient reasoning trace.

---

</details>

<details>
<summary><strong>Example 7: Sports Commentary Extraction (DROP) - All Correct ✅</strong></summary>

---

**Question:**

> In week 11, the Lions hosted their NFC North Division rivals the Green Bay Packers. The only scoring play of the first quarter was when Detroit's Jason Hanson kicked a 30-yard field goal. The Packers got on the board in the second quarter when Jermichael Finley caught a 20-yard touchdown pass from Aaron Rodgers. The Lions responded with a 1-yard touchdown run by Mikel Leshoure. In the third quarter, the Packers' M. D. Jennings intercepted a Matthew Stafford pass and ran it back 72 yards for a touchdown. The Lions responded with a Calvin Johnson 25-yard touchdown pass from Matthew Stafford. The Lions scored their final points in the fourth quarter when Jason Hanson kicked a 27-yard field goal. The Packers then scored a touchdown when Randall Cobb caught a 22-yard pass from Aaron Rodgers, and capped off their win with a 39-yard field goal by Mason Crosby. How many total points were scored in the first quarter? Provide ONLY a concise final answer, with no explanation, no punctuation, and no other text.

**Ground Truth:** 3

---

| Human-Readable CoT | Cipher | State Machine |
|--------------------|--------|---------------|
| Reasoning: In the first quarter, only two field goals were scored — 30-yard and 27-yard — but only the 30-yard one was in the first quarter. So only 3 points (field goal) were scored. | α: κ30-yard FG → 3 points | S0[1 FG = 3 pts]✓ |
| ✅ **Prediction: 3** | ✅ **Prediction: 3** | ✅ **Prediction: 3** |
| 📏 51 tokens | 📏 12 tokens | 📏 12 tokens |

> 📊 **Comments:** For reference, the SFT-only standard CoT model produced a reasoning trace of 142 tokens. The SFT-only Cipher and State Machine models produced 15 and 21 tokens respectively.

---

</details>

<details>
<summary><strong>Example 8: Statistical Extraction (DROP) - Only Standard CoT Correct ⚠️</strong></summary>

---

**Question:**

> There were 664,594 households out of which 24.35% had children under the age of 18 living with them, 46.28% were Marriage living together, 11.68% had a female householder with no husband present, and 37.40% were non-families. 30.11% of all households were made up of individuals and 14.70% (4.02% male and 10.68% female) had someone living alone who was 65 years of age or older. The average household size was 2.39 and the average family size was 2.97. How many in percent of households were families? Provide ONLY a concise final answer, with no explanation, no punctuation, and no other text.

**Ground Truth:** 62.60

---

| Human-Readable CoT | Cipher | State Machine |
|--------------------|--------|---------------|
| Reasoning: Households that are families are those that are not non-families. Given 37.40% were non-families, the remaining percentage were families: 100% - 37.40% = 62.60%. | κ24.35% → κ46.28% → κ11.68% → κ37.40% → κ30.11% → κ14.70% → μ=100−30.11−14.70=55.19 | S0[avg family size=2.97, avg household=2.39] → S1[family ≠ household] → S2[⊛ 2.97>2.39] ⟹ S3[✓family proportion=29.7%] |
| ✅ **Prediction: 62.60** *(only correct)* | ❌ **Prediction: 55.19** | ❌ **Prediction: 29.7** |
| 📏 59 tokens | 📏 72 tokens | 📏 59 tokens |

> ⚠️ **Comments:** For reference, the SFT-only standard CoT model produced a reasoning trace of 188 tokens. Both non-standard CoT struggled with information bombardment.

---

</details>

---
### Training Dynamics

Figure 6 below illustrates RL training trajectories for standard CoT on both datasets. The most salient pattern is the rapidity of compression: across both GSM8K and DROP, the majority of token reduction occurs within the first 100 training steps. This suggests that RL quickly identifies redundant reasoning patterns—repeated phrases, unnecessary elaborations, verbose transitions—and learns to eliminate them while preserving the core logical structure necessary for correct answers. Interestingly, DROP training exhibits markedly smoother convergence than GSM8K, with more stable reward curves and more consistent reasoning length trajectories. This stability difference may reflect DROP's simpler reasoning requirements, as we noted in §3.1 when observing the small gap between no-reasoning and with-reasoning baselines. Alternatively, Qwen3-30B's synthetic DROP reasoning may have contained more easily identifiable "fluff"—generic comprehension statements that add tokens without contributing to extractive or numerical reasoning—which RL could efficiently prune early in training.

#### Figure 6: RL Training Dynamics for Standard CoT Reasoning, GSM8K (left) and DROP (right). Test Sample Size n = 20.

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="results/plots/rl_SFT_gsm8k_gsm8k_32_ppo_0.0002_300.png" width="45%">
  <img src="results/plots/rl_SFT_drop_drop_32_0.0005_ppo_0.0002_700.png" width="45%">
</div>

These results raise a provocative question about the necessity of compressed representations. If RL can reduce human-readable reasoning by 50-83% while maintaining or even improving accuracy, the additional complexity of introducing specialized symbolic grammars may be unnecessary for many applications. Standard CoT offers inherent advantages: it requires no synthetic data generation for compression schemes, maintains human interpretability throughout training, and as our results demonstrate, compresses effectively under length-aware RL. While cipher and state machine achieve lower absolute token counts, the engineering overhead of designing, generating, and validating symbolic representations may not justify the marginal efficiency gains, particularly given their accuracy costs on more complex reasoning tasks like GSM8K.

---
### Ablation Studies

Due to computational constraints, our RL ablations are less systematic than those conducted for SFT. However, we performed targeted experiments to understand the stability requirements for different reasoning formats, as the challenges we encountered during initial training runs proved informative about the brittleness of RL optimization for reasoning tasks. Figure 7 compares reward trajectories across different configurations.

#### Figure 7: RL Reward Comparison on GSM8K

<p align="center">
<img src="results/plots/gsm8k_rl_ablation_reward_sweep.png" width="50%">
</p>

Objective clipping proved essential for preventing catastrophic failures. Our initial experiments without clipping (green line) resulted in complete reward collapse around step 100, likely due to high-variance reward signals where poor-quality traces produce destructive policy updates. Implementing PPO-style objective clipping [^11] stabilized standard CoT training at LR 2e-4 (blue line). 

However, even with clipping enabled, we found that LR 2e-4 remained unstable for cipher and state machine models, with preliminary runs showing early divergence similar to the green curve. Only by reducing the learning rate to 1e-4 did we achieve stable training for compressed representations (cyan and pink lines, Figure 7), though these runs exhibit noisier reward trajectories than standard CoT throughout optimization. This heightened sensitivity to learning rate echoes our SFT findings in §4.1, where compressed reasoning required more conservative optimization than human-readable formats. The pattern suggests a fundamental brittleness in compressed representations: because symbolic tokens carry precise semantic meaning (e.g., α = compute, S0 = initial state), incorrect generation or token substitution can more readily derail entire reasoning chains compared to natural language, where partial correctness and semantic redundancy provide robustness against small errors.


Budget constraints prevented systematic exploration of alternative $\ell_{\max}$ values or reward formulations. More aggressive length penalties tuned for compressed reasoning, or non-linear penalty functions, might unlock additional efficiency gains. Our single-epoch training may also not have reached full convergence, particularly for compressed models where conservative learning rates slow optimization.


# **5. Limitations and Future Work**

We identify and outline a few directions for future work here:

**Isolating the "Cardinality" Effect.** Our compressed representations preserve logical structure while reducing tokens. To test whether reasoning benefits arise purely from token presence rather than semantic content—as suggested by recent "filler token" hypotheses—future work could compare our structured compressions against semantically meaningless token sequences (e.g., repeated dots or pause tokens) matched for length. This would definitively isolate the contribution of reasoning structure versus computational depth.

**Cross-Task Transfer of Compressed Grammars.** We trained separate compression models for GSM8K and DROP. An open question is whether a unified compressed grammar can generalize across diverse reasoning tasks without per-task fine-tuning. Investigating task-agnostic compression strategies—perhaps through meta-learning or prompt-based grammar selection—could reveal whether efficient reasoning representations can be domain-general or must remain task-specific.

**Impact on General Language Capabilities.** Teaching models specialized reasoning syntax may degrade performance on standard conversational tasks, as compressed tokens occupy parameter capacity and attention. Future work should evaluate whether compressed reasoning creates a capability tradeoff: does optimizing for efficient task-specific reasoning compromise general linguistic competence? If so, understanding this tradeoff's severity and exploring mitigation strategies (e.g., adapter-based isolation) becomes critical for practical deployment.

**Minimal Supervision for Grammar Discovery.** Our approach requires supervised fine-tuning on synthetic compressed traces before reinforcement learning. DeepSeek-R1 noted that cold-start supervision was necessary in that it prevents formatting chaos where languages and formatting structures mix uncontrollably. However, this chaos might represent an opportunity: with sufficiently robust RL reward signals, could models discover their own efficient reasoning grammars from minimal examples, or even zero supervision? Exploring this spectrum—from heavily supervised grammar induction to emergent self-organization—could reveal fundamental insights about how models internalize reasoning structure.

# **6. Conclusion**

This work demonstrates that Chain-of-Thought reasoning can be substantially compressed without catastrophic accuracy loss. Compressed symbolic representations—particularly state machines—achieve 95-97% of human-readable performance while reducing token usage by 50-82%. This validates our hypothesis: models can reason effectively in symbolic spaces unconstrained by natural language conventions.

Critically, reinforcement learning reveals that *verbosity itself* is the primary inefficiency. When RL optimizes natural language CoT for token efficiency—removing redundant explanations while preserving logical structure—it achieves 50-83% token reduction with maintained accuracy, approaching symbolic compression levels (54-57 tokens versus 32-44 for cipher). The resulting traces remain interpretable, demonstrating that conciseness and readability are not mutually exclusive. The inefficiency comes not from natural language per se, but from unnecessary elaboration, repetitive phrasings, and explanatory verbosity in synthetic training data.

This finding has practical implications. Compressed symbolic reasoning achieves lower absolute token counts but exhibits brittleness—optimization instability, degenerate loops, sensitivity to hyperparameters. RL-compressed natural language offers competitive efficiency (within 2×) while preserving interpretability and training stability. For applications requiring human oversight, this 2× overhead may be acceptable. For automated throughput-optimized systems where interpretability is unnecessary, hand-crafted symbolic compression offers marginal gains at substantial engineering cost.

Reasoning efficiency is achievable through multiple pathways—symbolic grammars, RL-optimized natural language, or hybrid approaches. The optimal choice depends on task complexity, interpretability requirements, and deployment constraints. What remains clear is that verbose explanatory reasoning, while useful for human pedagogy, is computationally inefficient—models can achieve comparable or better task performance with substantially more concise representations, whether symbolic or streamlined natural language.

# **7. References** 

[^1]: DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning,” arXiv preprint, 2025. [Online]. Available: https://arxiv.org/abs/2501.12948.

[^2]: J. Pfau, W. Merrill, and S. R. Bowman, “Let's Think Dot by Dot: Hidden Computation in Transformer Language Models,” arXiv preprint, 2024. [Online]. Available: https://arxiv.org/abs/2404.15758.

[^3]: S. Goyal et al., “Think before you speak: Training Language Models with Pause Tokens,” arXiv preprint, 2023. [Online]. Available: https://arxiv.org/abs/2310.02226.

[^4]: H. Pan et al., “Training Large Language Models to Reason in a Continuous Latent Space,” arXiv preprint, 2024. [Online]. Available: https://arxiv.org/abs/2412.06769.

[^5]: Z. Shen et al., “CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation,” arXiv preprint, 2025. [Online]. Available: https://arxiv.org/abs/2502.21074.

[^6]: Qwen Team, “Qwen3 Technical Report,” arXiv preprint, 2025. [Online]. Available: https://arxiv.org/abs/2505.09388.

[^7]: K. Cobbe et al., “Training Verifiers to Solve Math Word Problems,” arXiv preprint, 2021. [Online]. Available: https://arxiv.org/abs/2110.14168.

[^8]: D. Dua et al., “DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs,” arXiv preprint, 2019. [Online]. Available: https://arxiv.org/abs/1903.00161.

[^9]: Z. Shao et al., “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models,” arXiv preprint, 2024. [Online]. Available: https://arxiv.org/abs/2402.03300.

[^10]: Schulman, John and Thinking Machines Lab, "LoRA Without Regret", Thinking Machines Lab: Connectionism, Sep 2025.

[^11]: J. Schulman et al., “Proximal Policy Optimization Algorithms,” arXiv preprint, 2017. [Online]. Available: https://arxiv.org/abs/1707.06347.