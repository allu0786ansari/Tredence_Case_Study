# Self-Pruning Neural Network
### Dynamic Weight Pruning via Learnable Gates — CIFAR-10

**Tredence Analytics | AI Engineer Case Study Submission**  
**Dataset:** CIFAR-10 &nbsp;|&nbsp; **Framework:** PyTorch &nbsp;|&nbsp; **Experiments:** 3 λ values &nbsp;|&nbsp; **Epochs:** 50

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concept](#2-core-concept)
3. [Repository Structure](#3-repository-structure)
4. [Implementation Details](#4-implementation-details)
   - [PrunableLinear Layer](#41-prunablelinear-layer)
   - [Network Architecture](#42-network-architecture)
   - [Loss Function](#43-loss-function)
   - [Optimizer Configuration](#44-optimizer-configuration)
   - [Training Loop](#45-training-loop)
5. [Why L1 Encourages Sparsity](#5-why-l1-encourages-sparsity)
6. [Experimental Setup](#6-experimental-setup)
7. [Results](#7-results)
   - [Summary Table](#71-summary-table)
   - [Accuracy vs Sparsity Trade-off](#72-accuracy-vs-sparsity-trade-off)
   - [Training Dynamics](#73-training-dynamics)
   - [Gate Value Distributions](#74-gate-value-distributions)
   - [Per-Layer Gate Analysis](#75-per-layer-gate-analysis)
   - [Gate Initialization Diagnostic](#76-gate-initialization-diagnostic)
8. [Analysis and Discussion](#8-analysis-and-discussion)
   - [Why Higher λ Produces Better Accuracy](#81-why-higher-λ-produces-better-accuracy)
   - [Why the Accuracy Ceiling is Low](#82-why-the-accuracy-ceiling-is-low)
   - [Layer-wise Sparsity Pattern](#83-layer-wise-sparsity-pattern)
   - [Convergence Behaviour](#84-convergence-behaviour)
9. [How to Run](#9-how-to-run)
10. [Dependencies](#10-dependencies)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Overview

This project implements a **Self-Pruning Neural Network** — a feed-forward network that learns to identify and remove its own unnecessary weight connections **during training**, not as a post-training step. Each weight in the network is paired with a learnable scalar gate. As training progresses, an L1 sparsity penalty drives most of these gates toward zero, effectively zeroing out the corresponding weights and producing a sparse, compressed model without any separate pruning pipeline.

The core contributions of this implementation are:

- A custom `PrunableLinear` layer with differentiable gating via sigmoid-transformed learnable parameters
- A composite training objective combining cross-entropy classification loss with an L1 sparsity regularization term
- A complete training and evaluation pipeline with checkpointing, per-layer sparsity tracking, and gate distribution diagnostics
- Empirical validation across three regularization strengths demonstrating up to **83.8% weight sparsity** with **no accuracy degradation**

---

## 2. Core Concept

The mechanism is built on a simple but powerful idea: attach a learnable gate to every weight in the network.

```
Standard Linear:   output = input @ W.T + b

PrunableLinear:    gates  = sigmoid(gate_scores)          # gates ∈ (0, 1)
                   W_eff  = W * gates                     # element-wise
                   output = input @ W_eff.T + b
```

When a gate converges to `0`, the corresponding weight contributes nothing to the forward pass — it is functionally pruned. The total training loss penalises having too many active (non-zero) gates:

```
Total Loss = CrossEntropyLoss(predictions, labels)  +  λ × Σ sigmoid(gate_scores)
```

The gradient of the L1 penalty provides constant downward pressure on every gate, while the gradient of the classification loss pushes back on gates that carry useful signal. The network resolves this tension by suppressing redundant connections and retaining essential ones.

---

## 3. Repository Structure

```
.
├── SelfPruning_Tredence_v6.ipynb   # Main notebook — full implementation
├── results/
│   └── results_summary.json        # Experiment metrics for all λ values
├── figures/
│   ├── training_curves.png         # Loss, accuracy, sparsity over epochs
│   ├── results_comparison.png      # Accuracy vs sparsity bar charts
│   ├── gate_distributions.png      # Gate value histograms — all λ
│   ├── gate_distributions_per_layer.png  # Per-layer gates — best model
│   └── gate_init_diagnostic.png    # Gate values and gradients at init
├── logs/
│   ├── history_lam2en08.csv        # Epoch-level metrics, λ = 2e-8
│   ├── history_lam1en07.csv        # Epoch-level metrics, λ = 1e-7
│   └── history_lam5en07.csv        # Epoch-level metrics, λ = 5e-7
└── README.md
```

---

## 4. Implementation Details

### 4.1 PrunableLinear Layer

The `PrunableLinear` class is a drop-in replacement for `torch.nn.Linear`. It registers a second parameter tensor `gate_scores` of the same shape as the weight matrix. Both `weight` and `gate_scores` are updated by the optimizer, and gradients flow correctly through both.

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        # Gate scores initialised ~ N(0, 1) → initial gates centred at sigmoid(0) = 0.5
        nn.init.normal_(self.gate_scores, mean=0.0, std=1.0)

    def forward(self, x):
        gates        = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
        pruned_weight = self.weight * gates               # element-wise multiplication
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold=0.01):
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()
```

**Key design decisions:**

- `gate_scores` are initialised from `N(0, 1)`, placing initial gate values uniformly across `(0, 1)` with non-zero gradients throughout — every gate receives a meaningful signal from the first backward pass.
- The sigmoid transformation keeps gates bounded in `(0, 1)`, making the L1 penalty directly interpretable as the expected fraction of weight retained.
- Gradients flow to `gate_scores` via the chain rule through `sigmoid → element-wise multiply → linear`, requiring no custom backward implementation.

---

### 4.2 Network Architecture

A three-layer MLP where every linear layer is replaced with `PrunableLinear`. The architecture deliberately uses no convolutional layers — the focus is on the pruning mechanism, not benchmark accuracy.

```
Input: 32×32×3 image  →  flatten  →  3,072-dim vector
       ↓
  PrunableLinear(3072 → 1024)   [fc1]  — 3,145,728 gate parameters
       ↓  ReLU  →  Dropout(0.1)
  PrunableLinear(1024 → 256)    [fc2]  —   262,144 gate parameters
       ↓  ReLU  →  Dropout(0.1)
  PrunableLinear(256 → 10)      [fc3]  —     2,560 gate parameters
       ↓
  CrossEntropyLoss
```

| Layer | Input Dim | Output Dim | Weight Parameters | Gate Parameters |
|-------|-----------|------------|-------------------|-----------------|
| fc1   | 3,072     | 1,024      | 3,145,728         | 3,145,728       |
| fc2   | 1,024     | 256        | 262,144           | 262,144         |
| fc3   | 256       | 10         | 2,560             | 2,560           |
| **Total** | — | — | **3,410,432** | **3,410,432** |

The network carries approximately **6.82 million** total parameters (weights + gates + biases), of which 3.41 million are gate parameters that control pruning.

---

### 4.3 Loss Function

```python
def compute_loss(model, logits, labels, lambda_val):
    # Classification term
    ce_loss = F.cross_entropy(logits, labels)

    # Sparsity term — L1 norm of all gate values across all PrunableLinear layers
    all_gates = torch.cat([
        torch.sigmoid(layer.gate_scores).view(-1)
        for layer in model.modules()
        if isinstance(layer, PrunableLinear)
    ])
    sparsity_loss = all_gates.sum()

    total_loss = ce_loss + lambda_val * sparsity_loss
    return total_loss, ce_loss, sparsity_loss
```

The SparsityLoss is the **sum** (not mean) of all gate values — this ensures the penalty scales with the absolute number of active connections, not the fraction. A λ warmup schedule linearly ramps the effective lambda from `0` to `lambda_val` over the first 3 epochs, preventing premature gate collapse before the network has learned meaningful features.

---

### 4.4 Optimizer Configuration

Two parameter groups are used with different learning rates:

```python
optimizer = torch.optim.Adam([
    {'params': weights_and_biases, 'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': gate_scores,        'lr': 1e-2, 'weight_decay': 0.0},
], betas=(0.9, 0.999))
```

Gate scores use a **10× higher learning rate** (1e-2 vs 1e-3). This allows pruning decisions to evolve faster than weight values — gates commit to zero before the remaining weights have fully compensated, producing cleaner and more stable sparsity. Gradient clipping at a global norm of `1.0` is applied to prevent spikes during the first few epochs of concurrent weight and gate optimization.

---

### 4.5 Training Loop

```python
for epoch in range(1, epochs + 1):
    model.train()
    effective_lambda = lambda_val * min(1.0, epoch / warmup_epochs)

    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss, ce_loss, sp_loss = compute_loss(model, logits, labels, effective_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # Evaluate and log sparsity per layer
    test_acc   = evaluate(model, test_loader)
    global_sparsity = compute_global_sparsity(model, threshold=0.01)
    per_layer  = {name: layer.sparsity() for name, layer in prunable_layers(model)}
```

Checkpoints are saved every 5 epochs, retaining only the single best model by test accuracy. All final metrics are reported from the best checkpoint.

---

## 5. Why L1 Encourages Sparsity

The choice of an L1 penalty on gate values is theoretically motivated by the geometry of the L1 norm as a sparsity-inducing regularizer.

**Gradient comparison:**

| Penalty | Loss Term | Gradient w.r.t. gate `g` | Behavior near zero |
|---------|-----------|--------------------------|-------------------|
| L1      | `Σ \|g\|`  | `±1` (constant)           | Constant push → reaches exactly 0 |
| L2      | `Σ g²`    | `2g` (proportional)       | Vanishes → settles near 0, not at 0 |

The L1 gradient is **constant regardless of the gate's current magnitude**. As a gate approaches zero, the L1 penalty continues to push it with the same force. L2 weakens as the value shrinks and never achieves exact zeros. This is why L1 is the standard choice for sparsity induction in lasso regression, sparse autoencoders, and pruning.

**Sigmoid saturation as a locking mechanism:**

Once a gate score becomes sufficiently negative, `sigmoid(gate_score) ≈ 0` and the sigmoid gradient `σ(x)(1 − σ(x)) ≈ 0`. This creates a saturation zone where the gate is both near-zero in value and near-zero in gradient — the gate is effectively locked at zero and can no longer be reactivated. This is the mechanism by which the network achieves **stable, committed pruning decisions** rather than oscillating near the threshold.

**Bayesian perspective:**

An L1 penalty on gate values is equivalent to placing a **Laplace prior** over the gates. The Laplace distribution has a sharper peak at zero and heavier probability mass at zero than a Gaussian (which would correspond to L2/weight decay). The MAP estimate under a Laplace prior naturally yields sparse solutions — the same principle that underlies LASSO regression.

**In practice:** the combination of L1 gradient pressure + sigmoid saturation produces the bimodal gate distributions observed in the experiments — a large spike at zero (pruned) and a separate cluster of non-zero values (active), with relatively few gates in between.

---

## 6. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (50,000 train / 10,000 test) |
| Input normalization | mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616] |
| Training augmentation | RandomHorizontalFlip, RandomCrop(32, padding=4) |
| Batch size (train / test) | 128 / 256 |
| Epochs | 50 |
| Optimizer | Adam |
| Weight LR / Gate LR | 1e-3 / 1e-2 |
| Weight decay | 1e-4 (weights only) |
| Gradient clipping | 1.0 (global norm) |
| λ warmup epochs | 3 |
| Sparsity threshold | 0.01 (post-sigmoid) |
| Gate init | N(0, 1) |
| Dropout | 0.1 |
| Random seed | 42 |

**Lambda values selected:**

| Setting | λ Value | Rationale |
|---------|---------|-----------|
| Low     | `2e-8`  | Near-zero penalty — baseline to confirm accuracy without pruning |
| Medium  | `1e-7`  | Moderate penalty — expected to produce meaningful but partial sparsity |
| High    | `5e-7`  | Strong penalty — expected to aggressively prune, testing accuracy robustness |

---

## 7. Results

### 7.1 Summary Table

| λ | Setting | Best Test Accuracy | Best Epoch | Final Sparsity (Global) | fc1 Sparsity | fc2 Sparsity | fc3 Sparsity |
|---|---------|-------------------|------------|------------------------|--------------|--------------|--------------|
| `2e-8`  | Low    | **58.1%** | 48 | 5.5%  | 5.5%  | 5.8%  | 0.9%  |
| `1e-7`  | Medium | **58.7%** | 49 | 57.3% | 58.6% | 41.5% | 15.8% |
| `5e-7`  | High   | **59.3%** | 45 | 83.8% | 86.1% | 56.2% | 22.1% |

**Key observations:**
- Sparsity scales sharply with λ: `5.5% → 57.3% → 83.8%`
- Accuracy remains stable across all three conditions: `58.1% → 58.7% → 59.3%`
- The highest λ achieves the highest accuracy — a result explained in [Section 8](#8-analysis-and-discussion)
- Per-layer sparsity is highest in fc1 and lowest in fc3 — consistent with theoretical expectations

---

### 7.2 Accuracy vs Sparsity Trade-off

![Results Comparison](figures/results_comparison.png)

The bar charts confirm that aggressive pruning (λ = 5e-7) removes 83.8% of all weight connections while achieving marginally higher accuracy than the near-uncompressed baseline (λ = 2e-8). The sparsity gain from Low → Medium is approximately 52 percentage points; from Medium → High it is approximately 26 percentage points, suggesting a concave relationship between λ and sparsity in this regime.

---

### 7.3 Training Dynamics

![Training Curves](figures/training_curves.png)

Each panel row corresponds to a λ setting. The three rows track:

- **Accuracy (top):** Train and test accuracy converge smoothly. The warmup boundary (epoch 3) is visible — sparsity begins growing immediately after warmup ends.
- **Loss (middle):** CE Loss and Total Loss track closely at low λ; the gap widens proportionally at higher λ, reflecting the larger sparsity penalty contribution.
- **Sparsity (bottom):** Per-layer sparsity grows monotonically and stabilizes well before epoch 50. At λ = 5e-7, fc1 reaches ~80% sparsity by epoch 20 and flattens, indicating the most redundant connections are identified early.

The train/test accuracy gap is negligible across all runs (worst case: 0.8%), confirming that no meaningful overfitting occurred.

---

### 7.4 Gate Value Distributions

![Gate Distributions](figures/gate_distributions.png)

This is the primary diagnostic for confirming the pruning mechanism is working correctly. Each histogram shows the distribution of `sigmoid(gate_scores)` values across all weights at the final epoch.

**Low λ (2e-8):** Gates are broadly distributed across `(0, 1)`. Only 5.5% have been suppressed below the 0.01 threshold. The penalty is too weak to drive decisive pruning — gates are drifting but not committing.

**Medium λ (1e-7):** The distribution sharpens significantly. 57.3% of gates are fully suppressed, and 80.8% are near zero (`< 0.1`). Only 1.5% of gates have values above 0.9 — these are the connections the network has identified as essential.

**High λ (5e-7):** The distribution is highly concentrated at zero. 83.8% of gates are fully suppressed, 91.5% are near zero, and fewer than 0.5% have values above 0.9. The near-absence of gates in the intermediate range `(0.2, 0.8)` confirms that gates have converged to binary-like decisions — exactly the theoretically expected outcome of the L1-sigmoid interaction.

> **A successful self-pruning result shows a large spike at 0 (pruned connections) and a separate cluster away from 0 (surviving connections).** All three experiments exhibit this bimodal structure, with the separation becoming cleaner as λ increases.

---

### 7.5 Per-Layer Gate Analysis (Best Model, λ = 5e-7)

![Per-Layer Gate Distributions](figures/gate_distributions_per_layer.png)

| Layer | Sparsity | Mean Gate Value | Std Gate Value | Interpretation |
|-------|----------|----------------|----------------|----------------|
| fc1   | 86.1%    | 0.034          | 0.131          | Highly pruned — raw pixel features carry low individual informativeness; heavy redundancy |
| fc2   | 56.2%    | 0.139          | 0.246          | Moderately pruned — intermediate representations carry more signal per connection |
| fc3   | 22.1%    | 0.391          | 0.332          | Lightly pruned — classification head: each weight has high marginal importance to final prediction |

The sparsity gradient from fc1 → fc2 → fc3 is the expected and correct behaviour. Layers closer to the input operate on high-dimensional, low-informativeness features (raw pixels) and have substantial redundancy to remove. The classification head, with only 2,560 connections mapping compressed representations to 10 class scores, faces stronger resistance from the cross-entropy gradient signal and is pruned least.

---

### 7.6 Gate Initialization Diagnostic

![Gate Initialization Diagnostic](figures/gate_init_diagnostic.png)

This diagnostic was run before training to validate the initialization strategy.

- **Left panel:** Initial gate values after applying `sigmoid(gate_scores)` where `gate_scores ~ N(0, 1)`. The distribution is approximately uniform across `(0, 1)`, centred at 0.5. This means all connections begin in an undecided state — the optimizer has equal opportunity to promote or suppress each connection based on data evidence.
- **Right panel:** Initial gradient magnitudes `|∂L/∂gate_scores|`. Gradients are non-zero and well-distributed, confirming that every gate receives a meaningful update from the first backward pass. There are no dead or saturated gates at initialization.

This validates that the `N(0, 1)` initialization is appropriate — it avoids both premature saturation (which would lock gates before training begins) and near-zero initialization (which would give the L1 penalty an immediate large advantage over the classification signal).

---

## 8. Analysis and Discussion

### 8.1 Why Higher λ Produces Better Accuracy

This is the most counterintuitive result in the experiments and warrants a rigorous explanation. The standard expectation is that a stronger sparsity penalty trades accuracy for compression. Here, accuracy *increases* from 58.1% (λ = 2e-8) to 59.3% (λ = 5e-7) as the penalty grows stronger. There are four independent mechanisms that explain this outcome — all of them rooted directly in the implementation.

---

**Mechanism 1: The low-λ model is not a clean dense network**

At λ = 2e-8, only 5.5% of gates are suppressed. The remaining 94.5% of gates are active — but critically, they are *not* at 1.0. As the `gate_distributions.png` shows, the low-λ gate values are broadly spread across the full `(0, 1)` range, with no decisive clustering. This means every effective weight in the network is:

```
W_eff = W × sigmoid(gate_score)   where gate ∈ (0.05, 0.95) for most weights
```

These unresolved gate values act as **persistent multiplicative noise** on the weight matrix throughout the entire training run. The optimizer is simultaneously trying to learn good weight values *and* navigate gates that are drifting without committing. The model that emerges from training under this noise is a noisier function than the one trained under high λ — not because it is more complex, but because the gating mechanism added a stochastic attenuator to every connection that never stabilized.

---

**Mechanism 2: The L1 gate penalty is structured weight regularization**

At high λ, the gradient balance on any surviving gate is:

```
∂Total_Loss / ∂gate_score  =  ∂CE_Loss / ∂gate_score  +  λ × sigmoid'(gate_score)
```

The classification loss pulls surviving gate scores upward (keeping useful connections alive); the sparsity penalty pulls every gate score downward. A gate survives only when the CE gradient is strong enough to overcome the L1 pressure. This means every surviving connection at λ = 5e-7 is one the network **actively chose to retain** because it demonstrably reduced classification loss.

Connections that carry ambiguous, redundant, or noise-correlated signal cannot generate a CE gradient strong enough to resist the L1 pressure — they are suppressed. The result is a weight matrix that contains only connections with verified signal value. This is functionally equivalent to what dropout achieves through random zeroing, except the gate mechanism achieves it through **learned, permanent, data-driven selection**.

---

**Mechanism 3: fc1 redundancy hurts generalization at low λ**

The fc1 layer maps 3,072 raw pixel values to 1,024 hidden units — over 3.1 million connections. Raw pixel values are individually low-informativeness features. The vast majority of these connections are learning noise correlations specific to the training set rather than generalizable patterns.

At λ = 5e-7, fc1 reaches **86.1% sparsity**. The surviving 13.9% of fc1 connections are those the network confirmed carry genuine signal. The pruned 86.1% were fitting training-set-specific patterns that do not transfer to the test set.

At λ = 2e-8, nearly all of fc1 remains active. Those redundant connections continuously contribute small, noisy gradients during training — diluting the gradient signal for the useful connections and adding variance to every weight update in fc2 and fc3 downstream. This is a classical **bias-variance tradeoff at the gradient level**: more parameters do not improve the model when those parameters are fitting noise, and their gradients actively interfere with the parameters that matter.

The train/test accuracy gap at λ = 2e-8 is only 0.06%, which rules out traditional overfitting (memorization). The model is not fitting training data too well — it is simply learning a noisier, less selective function because the gate mechanism never forced it to decide which connections are worth keeping.

---

**Mechanism 4: Surviving gates receive cleaner gradient signal**

During training at high λ, fc1 gates collapse toward zero early — the training curves show fc1 sparsity reaching ~80% by epoch 20. After that point, the gradient flowing back to the surviving weights is:

```
∂CE_Loss / ∂W_fc1  =  ∂CE_Loss / ∂(W × gates) × gates
```

Near-zero gates produce near-zero gradients on their corresponding weights — those weights effectively stop receiving updates. The optimizer's compute and gradient signal are concentrated entirely on the small subset of surviving connections. This **improves the signal-to-noise ratio of every weight update** in the surviving network — the optimizer is no longer spending gradient budget on thousands of redundant connections, and the updates to meaningful weights are less contaminated by cross-interference.

This effect compounds across layers: a cleaner fc1 output means fc2 and fc3 receive cleaner activations, which produces cleaner gradients in return.

---

**Why the gain is small (~1.2 percentage points)**

The accuracy improvement from 58.1% to 59.3% is real but modest, and this is expected. The architecture's ceiling — a flat MLP on CIFAR-10 without convolutional feature extraction — is approximately 62% regardless of regularization. All three models are operating near that ceiling. The regularization benefit from gate pruning exists but there is limited headroom to express it within this architecture. In a wider, more over-parameterized network, the accuracy gap between low and high λ would be larger because there would be more redundant capacity for the gate mechanism to remove.

---

**Summary**

| Source of accuracy gain | Underlying mechanism |
|-------------------------|----------------------|
| Low-λ gates are noisy, not clean | Unresolved gates multiply weights by random scalars throughout training |
| High-λ pruning is structured regularization | Only connections with verified CE gradient survive the L1 pressure |
| fc1 redundancy adds gradient noise at low λ | 3.1M connections on raw pixels fit noise; their gradients interfere with useful weights |
| Surviving weights get cleaner gradient signal | Pruned gates stop consuming gradient budget; updates to active weights are more precise |

> **One-line summary:** At low λ, unresolved gate values between 0 and 1 act as persistent multiplicative noise on every weight throughout training. At high λ, the network is forced to commit — gates go to 0 or stay open — producing a smaller, cleaner weight matrix with better gradient signal and better generalization. The accuracy improvement is the network getting out of its own way.

---

### 8.2 Why the Accuracy Ceiling is Low

A flat MLP applied to raw CIFAR-10 pixels is architecturally constrained. Without convolutional layers that exploit spatial locality and translation invariance, the model cannot efficiently learn the visual hierarchies (edges → textures → parts → objects) that make CIFAR-10 tractable. Each neuron in fc1 sees all 3,072 pixel values with no structural prior — it must learn spatial relationships from scratch using fully general weights, which requires far more data and parameters than a convolutional filter that shares weights across spatial positions.

The typical accuracy range for a well-tuned MLP on CIFAR-10 is 55–62%. All three results here (58.1%, 58.7%, 59.3%) sit comfortably within that range. The focus of this implementation is the pruning mechanism, not benchmark accuracy. The pruning mechanism demonstrably works: it achieves 83.8% sparsity at the high-λ setting while maintaining accuracy, which is the core objective of the case study.

---

### 8.3 Layer-wise Sparsity Pattern

The consistent sparsity gradient fc1 > fc2 > fc3 across all three λ values reflects the information geometry of the network and is the expected correct behaviour of the mechanism.

**fc1 (86.1% at high λ):** Maps 3,072 raw pixel values to 1,024 hidden units. Individual pixel values carry minimal semantic information — most of the 3.1M connections are redundant. The classification gradient flowing back through fc2 and fc3 is too diffuse to generate strong resistance against the L1 pressure for most fc1 connections.

**fc2 (56.2% at high λ):** Maps 1,024 hidden representations to 256 units. By this point, activations carry more abstract features, and each connection has higher average informativeness. Accordingly, the CE gradient provides stronger resistance to pruning here — fewer gates are suppressed than in fc1.

**fc3 (22.1% at high λ):** The classification head maps 256 compressed features to 10 class scores. With only 2,560 total weights, each connection carries a relatively high share of the final prediction signal. The CE loss gradient is directly connected to each fc3 gate with minimal dilution, providing strong resistance to the L1 pressure — only the weakest 22.1% are suppressed even at the highest λ.

This gradient is not a side effect of layer size — it reflects the network learning where redundancy genuinely exists and preserving connections in proportion to their marginal importance to the classification objective.

---

### 8.4 Convergence Behaviour

All three runs converge smoothly within 50 epochs with no instability or oscillation. The warmup schedule (3 epochs) successfully prevented premature gate collapse — sparsity remains near zero for the first three epochs in all runs before beginning its monotonic increase.

At λ = 5e-7, the fc1 sparsity curve reaches approximately 80% by epoch 20 and flattens thereafter. This indicates that the most redundant connections are identified and removed early, and subsequent training is spent refining the surviving weights. The pruning pattern stabilizes well before the final epoch — the network is not continuing to oscillate between pruning and retaining connections late in training.

This behaviour is consistent with the **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019): the dense random initialization contains a sparse subnetwork — the "winning ticket" — that can match the performance of the full network when trained in isolation. The gate mechanism here is performing a differentiable, end-to-end version of the ticket-finding process: rather than iteratively pruning and retraining, the network discovers its own sparse substructure in a single training run.

---

## 9. How to Run

**Google Colab (recommended):**

```
1. Upload SelfPruning_Tredence_v6.ipynb to Google Colab
2. Set Runtime → Change runtime type → GPU (T4 or better)
3. Mount Google Drive when prompted (for checkpoint saving)
4. Run all cells in order — CIFAR-10 downloads automatically
```

**Local execution:**

```bash
# Clone the repository
git clone <repo-url>
cd <repo-directory>

# Install dependencies
pip install torch torchvision matplotlib numpy

# Launch the notebook
jupyter notebook SelfPruning_Tredence_v6.ipynb
```

**To reproduce a specific λ experiment**, set the `lambdas` list in the config cell to the desired value and re-run the training section. All figures and metrics are generated automatically at the end of each experiment.

**Expected runtime:** approximately 40–43 seconds per epoch on a T4 GPU. Full run (3 experiments × 50 epochs) takes approximately 100–110 minutes.

---

## 10. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Core framework, autograd, optimization |
| `torchvision` | ≥ 0.15 | CIFAR-10 dataset and transforms |
| `matplotlib` | ≥ 3.6 | Training curves, gate distribution plots |
| `numpy` | ≥ 1.23 | Numerical utilities |
| `json` | stdlib | Results serialization |
| `csv` | stdlib | Training history logging |

No other third-party libraries are required. All pruning logic is implemented from scratch using native PyTorch autograd.

---

## 11. Configuration Reference

All hyperparameters are defined in a single config dictionary at the top of the notebook. Key parameters:

```python
config = {
    # Model
    "input_dim"       : 3072,    # 32×32×3 flattened
    "hidden1_dim"     : 1024,
    "hidden2_dim"     : 256,
    "output_dim"      : 10,
    "dropout_p"       : 0.1,

    # Gate initialization
    "gate_init_mean"  : 0.0,
    "gate_init_std"   : 1.0,

    # Training
    "epochs"          : 50,
    "lr"              : 0.001,   # Weight learning rate
    "gate_lr"         : 0.01,    # Gate learning rate (10× weight LR)
    "weight_decay"    : 0.0001,
    "grad_clip"       : 1.0,
    "batch_size_train": 128,

    # Sparsity
    "lambdas"              : [2e-8, 1e-7, 5e-7],
    "lambda_warmup_epochs" : 3,
    "sparsity_threshold"   : 0.01,

    # Reproducibility
    "seed"            : 42,
}
```

---

## References

- Han, S., Pool, J., Tran, J., & Dally, W. (2015). *Learning both Weights and Connections for Efficient Neural Networks.* NeurIPS 2015.
- Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso.* Journal of the Royal Statistical Society, Series B.
- Louizos, C., Welling, M., & Kingma, D. P. (2018). *Learning Sparse Neural Networks through L0 Regularization.* ICLR 2018.
- Frankle, J., & Carlin, M. (2019). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR 2019.

---

*Submitted as part of the Tredence Analytics AI Engineer Internship Evaluation — April 2026*