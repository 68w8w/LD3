# LD3-D: Learning to Discretize Discrete Diffusion
## Optimal Time Scheduling for Discrete State-Space Generative Models

---

## 1. Motivation and Problem Statement

Discrete diffusion models (D3PM, SEDD, MDLM, Discrete Flow Matching) have emerged as
powerful alternatives to autoregressive models for generating discrete data such as text,
discrete images, molecular structures, and code. These models formulate the generative
process as a continuous-time Markov chain (CTMC), where a forward corruption process
gradually destroys structure and a learned reverse process reconstructs it.

**The Sampling Bottleneck.** At inference time, the continuous-time reverse process must be
discretized into a finite number of steps. Current practice uses heuristic schedules
(uniform in time, linear in mask-rate, etc.), but the *optimal placement* of these timesteps
remains an open question for discrete diffusion.

**LD3 for Continuous Diffusion.** LD3 (ICLR 2025 Oral) demonstrated that learning the
optimal time discretization for continuous diffusion ODEs yields significant quality
improvements without any neural network retraining -- only N+1 scalar parameters are
optimized. However, LD3 is fundamentally tied to Gaussian noise processes and continuous
ODE/SDE solvers, making it inapplicable to discrete state spaces.

**This Work.** We propose **LD3-D**, the first principled framework for learning optimal time
discretization schedules for discrete diffusion models. Our approach:

1. Derives explicit per-step error bounds for CTMC-based discrete diffusion that decompose
   the global sampling error into local contributions dependent on step placement
2. Proposes a lightweight, solver-agnostic optimization that learns N+1 parameters via
   differentiable probability-flow tracking
3. Is compatible with all major discrete diffusion paradigms (absorbing/mask, uniform,
   discrete flow matching) and all major samplers (tau-leaping, theta-trapezoidal, analytic)

---

## 2. Background: Discrete Diffusion as CTMCs

### 2.1 State Space and Notation

Let X = {1, 2, ..., K} be the discrete state space (vocabulary of size K).
For sequences of length L, the joint state space is X^L.
We denote the data distribution as p_data over X^L.

### 2.2 Forward Process

The forward corruption process is defined by a rate matrix R_t in R^{K x K}:

- Off-diagonal: R_t(i,j) >= 0 for i != j (rate of transitioning from state i to j)
- Diagonal: R_t(i,i) = -sum_{j != i} R_t(i,j) (rows sum to zero)
- Time-dependent scaling: R_t = sigma(t) * R_base

The transition probability matrix satisfies the Kolmogorov forward equation:

    dQ_{t|s}/dt = Q_{t|s} * R_t,    Q_{s|s} = I

The marginal at time t factorizes per position (conditional on x_0):

    q_t(x^l | x_0^l) = e_{x_0^l}^T * Q_{t|0}

**Common Forward Processes:**

**(a) Absorbing (Mask) Diffusion** (D3PM, MDLM):

    R_base^abs(i, j) = { 1  if j = [MASK] and i != [MASK]
                        { 0  otherwise

Each token transitions to the mask state [MASK] at rate sigma(t). The transition matrix
has closed form:

    Q_{t|0}^abs(i, j) = { (1 - beta_t)           if j = i, i != [MASK]
                         { beta_t                  if j = [MASK], i != [MASK]
                         { 1                       if j = i = [MASK]

where beta_t = 1 - exp(-integral_0^t sigma(s) ds).

**(b) Uniform Diffusion** (D3PM, SEDD):

    R_base^uni = (1/K) * 1*1^T - I

Each token transitions uniformly to any other token. Transition matrix:

    Q_{t|0}^uni(i, j) = { (1 - beta_t) + beta_t/K    if j = i
                         { beta_t/K                    if j != i

**(c) Discrete Flow Matching (DFM)** (Gat et al., 2024):

    p_{t|0,1}(x | x_0, x_1) = (1-t) * delta(x, x_0) + t * delta(x, x_1)

Interpolates between source x_0 and target x_1 in probability space.

### 2.3 Reverse Process

The time-reversed CTMC has rate matrix (detailed balance):

    R_bar_t(x, y) = R_t(y, x) * p_t(y) / p_t(x),    x != y

The **concrete score** (ratio) is defined as:

    s_t(y | x) = p_t(y) / p_t(x)

A neural network s_theta(x_t, t) is trained to approximate this ratio via:
- **Score entropy loss** (SEDD): L_SE = E[sum_{y != x} s_theta(y|x) - s*(y|x) * log s_theta(y|x)]
- **Denoising cross-entropy** (MDLM): L_CE = -E[log p_theta(x_0 | x_t, t)]

### 2.4 Sampling (Time Discretization)

Given a schedule tau = (t_N = T > t_{N-1} > ... > t_0 = 0) and step sizes h_i = t_i - t_{i-1}:

**tau-Leaping (First Order):**

For each step i = N, ..., 1 and each position l:
1. Compute reverse rate: R_hat_t_i(x, y) = R_t_i(y, x) * s_theta(y | x_{t_i}, t_i)
2. Transition probability: p(x_{t_{i-1}}^l = y | x_{t_i}^l = x) = delta(x,y) + R_hat_t_i(x,y) * h_i

**Exact (Matrix Exponential):**

    p(x_{t_{i-1}} | x_{t_i}) = e_{x_{t_i}}^T * exp(R_hat_t_i * h_i)

**theta-Trapezoidal (Second Order)** (Zhao et al., 2025):

    p(x_{t_{i-1}} | x_{t_i}) ~ (I - theta*h_i*R_hat_{t_{i-1}})^{-1} * (I + (1-theta)*h_i*R_hat_{t_i}) * e_{x_{t_i}}

---

## 3. Theoretical Framework: Error Decomposition

### 3.1 Global Error Bound for tau-Leaping

**Theorem 1 (Per-Step Error Decomposition).**
Let p* denote the distribution from the exact continuous-time reverse process and p_tau^N
the distribution from N-step tau-leaping with schedule tau. Under standard regularity
assumptions on s_theta, the KL divergence decomposes as:

    D_KL(p* || p_tau^N) <= sum_{i=1}^{N} E_i(h_i, t_i)

where the per-step error E_i has two components:

    E_i(h_i, t_i) = E_i^disc(h_i, t_i) + E_i^score(h_i, t_i)

**(a) Discretization Error:**

    E_i^disc(h_i, t_i) = (h_i^2 / 2) * d * E_{x ~ p_{t_i}} [ ||dR_bar_t/dt|_{t=t_i}||_FS^2 / ||R_bar_{t_i}||_FS ]

where d is the sequence length, and ||.||_FS is the "Fisher-Stein" operator norm defined as:

    ||A||_FS = sum_{y != x} |A(x,y)|^2 / R_bar_t(x,y)

This arises from the Baker-Campbell-Hausdorff expansion of exp(R*h) vs. I + R*h.

**(b) Score Estimation Error:**

    E_i^score(h_i, t_i) = h_i * d * epsilon_score(t_i)

where epsilon_score(t_i) = E_{x ~ p_{t_i}}[L_SE(s_theta, s*; x, t_i)] is the expected
score entropy loss at time t_i.

**Proof Sketch.** We use the Girsanov-type change of measure for CTMCs (adapted from
Zhang et al., 2024; Conforti et al., 2025). The KL between path measures of two CTMCs
with rate matrices R and R' over interval [t_{i-1}, t_i] is:

    D_KL(P_R || P_{R'}) = integral_{t_{i-1}}^{t_i} E_{x~p_t^R}[sum_{y!=x} R(x,y)*log(R(x,y)/R'(x,y)) - R(x,y) + R'(x,y)] dt

For tau-leaping, R' is the piecewise-constant approximation R'_t = R_hat_{t_i} over
[t_{i-1}, t_i]. Taylor expanding around t_i and using the score entropy structure yields
the stated bound.

### 3.2 Optimal Schedule (Closed Form)

**Corollary 1.** Given a fixed NFE budget N and assuming epsilon_score(t) varies slowly
relative to the discretization error, the optimal schedule minimizes:

    min_{h_1,...,h_N} sum_i (h_i^2 * C_i + h_i * D_i)
    s.t. sum_i h_i = T, h_i > 0

where C_i = (d/2) * E[||dR_bar/dt||_FS^2 / ||R_bar||_FS]|_{t_i} and D_i = d * epsilon_score(t_i).

By Lagrange multipliers, the KKT conditions give:

    h_i* = max(0, (mu - D_i) / (2 * C_i))

where mu is chosen so that sum_i h_i* = T.

**Intuition:** More steps should be allocated where:
1. C_i is large: the reverse rate matrix changes rapidly (high curvature in probability space)
2. D_i is large: the score network has high estimation error

For absorbing diffusion, C_i is typically largest near t=0 (when most tokens are still
masked and the model must make critical decisions) and near t=T (when the distribution
transitions from pure noise). This explains the empirically observed U-shaped allocation.

### 3.3 Rate of Change Analysis for Common Processes

**Absorbing Diffusion:**

    sigma(t) = -log(1 - t)  =>  dR_bar/dt ~ sigma'(t) * R_base * s_theta + sigma(t) * R_base * ds_theta/dt

The first term diverges as t -> 1 (sigma'(t) = 1/(1-t)), suggesting finer steps near T.
The score ds_theta/dt changes most rapidly near t=0 (critical denoising region).

**Uniform Diffusion:**

    Similar analysis shows C_i peaks at intermediate t where the "confusion" is maximal
    (tokens are partially corrupted but not yet fully randomized).

### 3.4 Error Bound for Higher-Order Methods

For the theta-Trapezoidal method, the per-step error improves to:

    E_i^disc(h_i, t_i) = O(h_i^3) * C_i'

The optimal schedule then satisfies h_i* ~ C_i'^{-1/3} / sum_j C_j'^{-1/3} * T.

---

## 4. LD3-D Algorithm

### 4.1 Overview

LD3-D learns the optimal timestep schedule tau = {t_i}_{i=0}^N for a pre-trained discrete
diffusion model using only N+1 learnable scalar parameters, without retraining any neural
networks.

### 4.2 Schedule Parameterization

Following LD3, we use a differentiable, monotonic parameterization:

    theta_1 in R^{N+1}  (primary weights)
    theta_2 in R^{N+1}  (perturbations)

    w = softmax(theta_1)                    -- non-negative weights summing to 1
    c = cumsum(w)                           -- monotonically increasing in [0, 1]
    t_steps_1 = c * (T - epsilon) + epsilon -- mapped to [epsilon, T]

    max_move = min_gap(t_steps_1) * window_rate
    t_steps_2 = t_steps_1 + clamp(theta_2, -max_move, max_move) * mask

### 4.3 Differentiable Proxy Objective

**Key Challenge:** Discrete sampling involves categorical decisions that break gradient flow.

**Solution: Probability-Flow Tracking.** Instead of sampling discrete tokens, we track the
full probability distribution through the reverse process:

For each position l = 1, ..., L:
    p_{t_i}^l in Delta^K  (probability simplex)

The tau-leaping update in probability space is:

    p_{t_{i-1}}^l = p_{t_i}^l * (I + R_hat_{t_i} * h_i)

where R_hat_{t_i} is computed from the score model evaluated at the *expected* input:

    x_soft_{t_i} = sum_k k * p_{t_i}^l(k)  (soft embedding)

This is fully differentiable! No Gumbel-Softmax needed for the schedule optimization,
since we optimize the schedule, not the tokens.

**Loss Functions:**

(a) For discrete tokens decoded to continuous space (VQ-VAE, images):
    L = LPIPS(Decode(x_tau), Decode(x_teacher)) + lambda * CE(p_tau, p_teacher)

(b) For text generation:
    L = CE(p_tau^{(0)}, x_teacher) = -sum_l log p_tau^{(0)}(x_teacher^l)

(c) Theory-motivated proxy:
    L = sum_i h_i^2 * C_hat_i(tau)

    where C_hat_i is estimated from the score model's behavior at t_i.

### 4.4 Training Procedure

**Phase 0: Teacher Data Generation**
- Run pre-trained model with M >> N steps (e.g., M=1024) using baseline schedule
- Store: {z_j, x_j^teacher, [p_j^teacher]}_{j=1}^{J}
- Only J = 25-50 samples needed (following LD3)

**Phase 1: Coarse Schedule Learning (params1 only)**
- Freeze theta_2 = 0
- Optimize theta_1 via RMSprop with momentum
- 2-5 rounds of training
- Learning rate: 0.005, decay on plateau

**Phase 2: Fine-tuning with Perturbations (params1 + params2)**
- Unfreeze theta_2
- Optimize both with separate learning rates
- 5+ rounds
- SGD for theta_2 with lr ~ 0.1/N

### 4.5 Computational Cost

- Schedule parameters: 2*(N+1) scalars (negligible memory)
- Per-training-step: One forward pass through score model + probability tracking
- Total training: ~100-500 forward passes (minutes on single GPU)
- No neural network gradients needed (only through schedule parameters)

---

## 5. Connections to Existing Work

### 5.1 vs. LD3 (Tong et al., ICLR 2025)

LD3 operates on continuous Gaussian diffusion with ODE solvers. LD3-D extends the
core idea to discrete CTMCs with fundamentally different:
- Error analysis (matrix exponential vs. ODE truncation)
- Differentiable path (probability flow vs. continuous ODE)
- Noise structure (transition matrices vs. alpha/sigma schedules)

### 5.2 vs. LSD/LSD+ (Liu et al., 2025)

LSD learns *sampler coefficients* (how to transition) via distillation from a teacher's
score trajectory. LD3-D learns *timestep placement* (when to evaluate). Key differences:

| Aspect | LSD/LSD+ | LD3-D |
|--------|----------|-------|
| What's learned | Sampler coefficients + schedule | Schedule only |
| Training signal | Score trajectory matching | Reconstruction loss |
| #Parameters | O(N * sampler_params) | O(N) scalars |
| Requires | Teacher trajectory access | Teacher outputs only |
| Solver-agnostic | No (specific to tau-leaping) | Yes (any solver) |

**LD3-D and LSD are orthogonal and composable:** first optimize the schedule with LD3-D,
then optimize the sampler coefficients with LSD for the learned schedule.

### 5.3 vs. Fast Solvers (Zhao et al., NeurIPS 2025)

Fast Solvers develop higher-order methods (theta-Trapezoidal) that reduce per-step error.
LD3-D optimizes step placement for any given solver. These are complementary:
LD3-D + theta-Trapezoidal > either alone.

### 5.4 vs. Convergence Theory (Zhang et al., 2024; Chen & Ying, 2024)

Theoretical works derive error bounds but do not optimize the schedule. LD3-D uses
these bounds as motivation and derives a practical learning algorithm.

---

## 6. Experimental Plan

### 6.1 Benchmarks

**(a) Text Generation**
- Models: MDLM (NeurIPS 2024), SEDD (ICML 2024 Best Paper)
- Datasets: text8, OpenWebText, LM1B
- Metrics: Perplexity (PPL), MAUVE score, generation speed
- NFE budgets: N in {8, 16, 32, 64, 128, 256}

**(b) Discrete Image Generation**
- Models: D3PM, VQ-Diffusion
- Datasets: CIFAR-10, ImageNet (VQ-VAE tokenized)
- Metrics: FID, IS, generation speed

**(c) Molecular Generation**
- Models: Discrete CTMC (Campbell et al., 2024)
- Datasets: QM9, ZINC
- Metrics: Validity, uniqueness, novelty

### 6.2 Baselines

1. **Uniform**: t_i = i/N * T
2. **Linear mask-rate**: beta(t_i) = i/N (for absorbing)
3. **Cosine**: Cosine schedule from improved DDPM
4. **Log-linear**: Uniform in log-rate space
5. **LSD/LSD+**: Learned sampler distillation
6. **DMN**: Analytical optimal from LD3's scipy optimizer (adapted)

### 6.3 Ablations

1. Schedule parameterization: softmax+cumsum vs. direct vs. spline
2. Loss function: CE vs. probability-flow KL vs. theory proxy
3. Training set size: 10, 25, 50, 100 samples
4. Phase 1 vs. Phase 2 contribution
5. Interaction with solver type: tau-leaping vs. theta-Trapezoidal vs. analytic
6. Absorbing vs. uniform forward process

### 6.4 Analysis

1. Visualize learned schedules: where do steps concentrate?
2. Compare with theoretical optimal from Corollary 1
3. Plot C_i and D_i profiles for different models/datasets
4. Quality vs. NFE Pareto curves
5. Transfer: does a schedule learned for N=32 help for N=16 or N=64?

---

## 7. Expected Contributions

1. **First principled framework** for optimal time discretization in discrete diffusion,
   with rigorous error bounds (Theorem 1) and closed-form optimal schedule (Corollary 1)

2. **Lightweight, solver-agnostic algorithm** that improves any discrete diffusion sampler
   by optimizing only O(N) scalar parameters, with no neural network retraining

3. **Differentiable probability-flow formulation** that enables gradient-based schedule
   optimization for discrete state spaces without Gumbel-Softmax relaxation

4. **Comprehensive empirical validation** across text, image, and molecular domains,
   demonstrating consistent improvements in quality-vs-compute tradeoffs

5. **Composability with existing methods** (LSD, Fast Solvers), showing orthogonal gains

---

## 8. Anticipated Reviewer Concerns and Rebuttals

**Q1: "This is just LD3 applied to discrete diffusion."**

A: The mathematical frameworks are fundamentally different. Continuous diffusion uses SDEs
with Gaussian perturbations; discrete diffusion uses CTMCs with transition matrices.
Our error analysis (Theorem 1) requires a Girsanov-type argument for CTMCs, not ODE
truncation analysis. The differentiable optimization requires probability-flow tracking
(a novel contribution), not continuous backpropagation through ODEs. The resulting
schedules have qualitatively different structure (U-shaped for absorbing vs. monotonic
for continuous VP).

**Q2: "How is this different from LSD+?"**

A: LSD+ jointly learns sampler coefficients and schedule via score-trajectory distillation.
LD3-D learns *only* the schedule via direct reconstruction optimization. This makes LD3-D:
(a) simpler (O(N) params vs. O(N*K) for LSD+), (b) solver-agnostic (LSD+ is tied to
tau-leaping), (c) composable with LSD+ for further gains. We demonstrate this
composability in experiments.

**Q3: "The probability-flow tracking is expensive for large vocabularies."**

A: Per-position probability vectors are K-dimensional. For typical text vocabularies (K~32K),
this is stored as [L, K] tensors. We process positions in parallel on GPU. The
computation is dominated by the score model forward pass (same as sampling), and the
schedule optimization requires only J=25-50 samples (total ~100-500 forward passes).
For very large K, we use a top-k approximation.

**Q4: "The error bound in Theorem 1 is loose."**

A: Yes, like all KL-based bounds for diffusion. However: (a) the bound correctly identifies
which terms depend on step placement, (b) the relative weighting of C_i across steps
guides the practical algorithm, and (c) we empirically validate that the theory-motivated
schedule correlates strongly with the learned optimal schedule (Section 6.4).

**Q5: "What if the score model is poor?"**

A: When epsilon_score dominates, the D_i term in the bound suggests uniform allocation
(all steps equally important). LD3-D degrades gracefully to uniform scheduling.
When the score model is strong (epsilon_score small), the C_i term dominates and
non-uniform scheduling provides the largest gains.

---

## 9. Implementation Architecture

### 9.1 New Files

```
LD3/
  discrete_noise_schedulers.py   -- CTMC forward processes
  discrete_samplers/
    __init__.py
    discrete_solver_base.py      -- Base class for discrete solvers
    tau_leaping.py               -- First-order tau-leaping
    analytic_sampler.py          -- Exact matrix exponential
  models/
    discrete_model_wrapper.py    -- Wrapper for MDLM/SEDD/D3PM
  discrete_trainer.py            -- LD3-D training loop
  discrete_gen_data.py           -- Teacher data generation
  configs/discrete/
    mdlm_text8.yml
    sedd_text8.yml
    d3pm_cifar10.yml
```

### 9.2 Design Principles

- Maintain LD3's modular architecture
- Same two-phase training strategy (params1 + params2)
- Same softmax+cumsum schedule parameterization
- Swap noise schedules, solvers, and loss functions for discrete versions
- Probability-flow tracking replaces continuous ODE backpropagation

---

## References

- Austin et al. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM). NeurIPS.
- Lou et al. (2024). Discrete Diffusion Modeling by Estimating Ratios of the Data Distribution (SEDD). ICML Best Paper.
- Sahoo et al. (2024). Simple and Effective Masked Diffusion Language Models (MDLM). NeurIPS.
- Gat et al. (2024). Discrete Flow Matching. NeurIPS.
- Campbell et al. (2024). Generative Flows on Discrete State-Spaces. ICML.
- Tong et al. (2025). Learning to Discretize Denoising Diffusion ODEs (LD3). ICLR Oral.
- Zhang et al. (2025). Convergence of Score-Based Discrete Diffusion Models. ICLR.
- Zhao et al. (2025). Fast Solvers for Discrete Diffusion Models. NeurIPS.
- Liu et al. (2025). Learnable Sampler Distillation for Discrete Diffusion Models.
- Conforti et al. (2025). Comprehensive Analysis of Discrete Diffusion Models. ICLR.
