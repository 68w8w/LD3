"""
Tau-Leaping sampler for discrete diffusion models.

First-order Euler method for CTMCs: approximate the transition over [t_{i-1}, t_i]
by holding the reverse rate matrix constant at R_hat_{t_i}.

    p(x_{t_{i-1}} = y | x_{t_i} = x) = delta(x,y) + R_hat_{t_i}(x,y) * h_i  for y != x
    p(x_{t_{i-1}} = x | x_{t_i} = x) = 1 + R_hat_{t_i}(x,x) * h_i

Reference:
  - Lou et al. (SEDD, ICML 2024)
  - Zhang et al. (Convergence of Score-Based Discrete Diffusion, ICLR 2025)
"""

import torch
import torch.nn.functional as F

from .discrete_solver_base import DiscreteSolver


class TauLeapingSolver(DiscreteSolver):
    """Tau-leaping (first-order) sampler for discrete diffusion.

    Supports:
      - Absorbing (mask) diffusion
      - Uniform diffusion
      - Discrete flow matching
    """

    def __init__(self, noise_schedule, diffusion_type='absorbing'):
        super().__init__(noise_schedule, diffusion_type)

    def step(self, x_t, t_cur, t_next, score_fn, t_eval=None, **kwargs):
        """One tau-leaping step from t_cur to t_next.

        Args:
            x_t: Token indices, shape [B, L]
            t_cur: Current time
            t_next: Target time (t_next < t_cur)
            score_fn: (x, t) -> score output
                For absorbing: [B, L, K] logits over clean tokens (x_0 prediction)
                For uniform/SEDD: [B, L, K] concrete score ratios
            t_eval: Time at which to evaluate score (for LD3-D perturbation)
        """
        if t_eval is None:
            t_eval = t_cur

        h = (t_cur - t_next).abs()  # step size

        if self.diffusion_type == 'absorbing':
            return self._step_absorbing(x_t, t_cur, t_next, h, score_fn, t_eval)
        elif self.diffusion_type == 'uniform':
            return self._step_uniform(x_t, t_cur, t_next, h, score_fn, t_eval)
        elif self.diffusion_type == 'flow':
            return self._step_flow(x_t, t_cur, t_next, h, score_fn, t_eval)
        else:
            raise ValueError(f"Unknown diffusion_type: {self.diffusion_type}")

    def _step_absorbing(self, x_t, t_cur, t_next, h, score_fn, t_eval):
        """Tau-leaping step for absorbing (mask) diffusion.

        For absorbing diffusion, the reverse rate matrix is:
            R_bar_t(MASK, y) = sigma(t) * p_theta(x_0 = y | x_t, t)  for y != MASK
            R_bar_t(y, MASK) = sigma(t) * (1 - sum_k p(x_0=k | x_t, t))  # negligible

        The key insight: only masked positions can change (unmask).
        """
        B, L = x_t.shape
        mask_idx = self.noise_schedule.mask_index
        device = x_t.device

        # Get score / x_0 prediction
        score_output = score_fn(x_t, t_eval)  # [B, L, K]

        # Identify masked positions
        is_masked = (x_t == mask_idx)  # [B, L]

        # Compute unmasking probability
        sigma_t = self.noise_schedule.rate_scalar(t_cur)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t = sigma_t.to(device)

        # p_theta(x_0 | x_t, t) from score output (logits -> probs)
        if score_output.dim() == 3:
            p_x0 = F.softmax(score_output, dim=-1)  # [B, L, K]
        else:
            p_x0 = score_output

        # Zero out the mask token probability (can't "unmask" to mask)
        p_x0_clean = p_x0.clone()
        p_x0_clean[:, :, mask_idx] = 0.0
        p_x0_clean = p_x0_clean / (p_x0_clean.sum(dim=-1, keepdim=True) + 1e-10)

        # Reverse rate for masked positions: R_bar(MASK, y) = sigma(t) * p(x_0=y)
        # Transition probability: p_unmask(y) = R_bar(MASK, y) * h = sigma(t) * h * p(x_0=y)
        unmask_rate = sigma_t * h
        # Clamp to ensure valid probability
        unmask_rate = torch.clamp(unmask_rate, max=0.99)

        # For masked positions: sample whether to unmask
        should_unmask = torch.bernoulli(
            unmask_rate * torch.ones(B, L, device=device)
        ).bool() & is_masked  # [B, L]

        # Sample new token from p_x0 for positions that unmask
        if should_unmask.any():
            new_tokens = torch.multinomial(
                p_x0_clean[should_unmask].float(), num_samples=1
            ).squeeze(-1)
            x_next = x_t.clone()
            x_next[should_unmask] = new_tokens
        else:
            x_next = x_t.clone()

        return x_next

    def _step_uniform(self, x_t, t_cur, t_next, h, score_fn, t_eval):
        """Tau-leaping step for uniform diffusion.

        For uniform diffusion with concrete score s_theta(y | x):
            R_bar_t(x, y) = sigma(t) / K * s_theta(y | x_t, t)  for y != x

        Transition: p(y | x) = delta(x,y) + h * sigma(t) / K * s_theta(y | x)
        """
        B, L = x_t.shape
        K = self.K
        device = x_t.device

        # Get concrete score ratios: s_theta(y | x) = p_t(y) / p_t(x)
        score_output = score_fn(x_t, t_eval)  # [B, L, K]

        sigma_t = self.noise_schedule.rate_scalar(t_cur)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t = sigma_t.to(device)

        # Reverse rate contributions: R_bar(x, y) * h
        # For uniform: R_t(y, x) = sigma(t) / K for y != x
        # R_bar(x, y) = R_t(y, x) * s_theta(y | x) = sigma(t) / K * s_theta(y | x)
        transition_rates = (sigma_t * h / K) * score_output  # [B, L, K]

        # Zero out self-transitions (will be set by normalization)
        x_one_hot = F.one_hot(x_t, K).float()  # [B, L, K]
        transition_rates = transition_rates * (1 - x_one_hot)

        # Clamp negative rates
        transition_rates = torch.clamp(transition_rates, min=0.0)

        # Total transition probability out of current state
        total_rate = transition_rates.sum(dim=-1, keepdim=True)  # [B, L, 1]

        # Stay probability
        stay_prob = torch.clamp(1.0 - total_rate, min=0.0)

        # Full transition distribution
        trans_probs = torch.cat([stay_prob, transition_rates], dim=-1)  # [B, L, K+1]
        # But we need [B, L, K] with self-transition
        trans_probs_normalized = transition_rates + stay_prob * x_one_hot
        trans_probs_normalized = trans_probs_normalized / (trans_probs_normalized.sum(dim=-1, keepdim=True) + 1e-10)

        # Sample
        x_next = torch.multinomial(
            trans_probs_normalized.reshape(B * L, K).float(), num_samples=1
        ).reshape(B, L)

        return x_next

    def _step_flow(self, x_t, t_cur, t_next, h, score_fn, t_eval):
        """Tau-leaping step for discrete flow matching.

        Similar to uniform but uses the flow velocity instead of score.
        """
        # For DFM, the score_fn returns the probability denoiser p(x_1 | x_t, t)
        return self._step_uniform(x_t, t_cur, t_next, h, score_fn, t_eval)

    def step_probability(self, p_t, t_cur, t_next, score_fn_soft, t_eval=None):
        """One tau-leaping step in probability space (differentiable).

        Instead of discrete tokens, operates on probability distributions.

        Args:
            p_t: Probability distributions, shape [B, L, K]
            t_cur, t_next: Times
            score_fn_soft: (p, t) -> score output for soft inputs
            t_eval: Evaluation time

        Returns:
            p_next: Updated probabilities, shape [B, L, K]
        """
        if t_eval is None:
            t_eval = t_cur

        h = (t_cur - t_next).abs()
        K = self.K
        device = p_t.device

        # Evaluate score on soft input
        score_output = score_fn_soft(p_t, t_eval)  # [B, L, K]

        if self.diffusion_type == 'absorbing':
            return self._step_prob_absorbing(p_t, h, score_output)
        elif self.diffusion_type == 'uniform':
            return self._step_prob_uniform(p_t, h, score_output, t_cur)
        else:
            return self._step_prob_uniform(p_t, h, score_output, t_cur)

    def _step_prob_absorbing(self, p_t, h, score_output):
        """Probability-space step for absorbing diffusion.

        Update rule for masked probability mass:
            p_{t-h}(y) = p_t(y) + h * sigma(t) * p_t(MASK) * p_theta(y | masked)  for y != MASK
            p_{t-h}(MASK) = p_t(MASK) - h * sigma(t) * p_t(MASK)
        """
        mask_idx = self.noise_schedule.mask_index
        sigma_t = self.noise_schedule.rate_scalar(torch.tensor(0.5))  # approximate

        p_x0 = F.softmax(score_output, dim=-1)  # [B, L, K]
        p_mask = p_t[:, :, mask_idx:mask_idx + 1]  # [B, L, 1]

        # Transfer probability from MASK to clean tokens
        unmask_flow = h * sigma_t * p_mask * p_x0  # [B, L, K]
        unmask_flow[:, :, mask_idx] = 0.0  # Don't flow to mask

        total_flow = unmask_flow.sum(dim=-1, keepdim=True)

        p_next = p_t + unmask_flow
        p_next[:, :, mask_idx:mask_idx + 1] = p_t[:, :, mask_idx:mask_idx + 1] - total_flow

        # Normalize
        p_next = torch.clamp(p_next, min=0.0)
        p_next = p_next / (p_next.sum(dim=-1, keepdim=True) + 1e-10)

        return p_next

    def _step_prob_uniform(self, p_t, h, score_output, t_cur):
        """Probability-space step for uniform diffusion.

        p_{t-h}(y) = p_t(y) + h * sum_{x != y} R_bar_t(x, y) * p_t(x) - h * sum_{y' != y} R_bar_t(y, y') * p_t(y)

        Simplified with concrete score:
        p_{t-h} = p_t * (I + h * R_bar_t)  [matrix-vector product per position]
        """
        K = self.K
        sigma_t = self.noise_schedule.rate_scalar(t_cur)

        # R_bar(x, y) = sigma(t)/K * score(y|x) for x != y
        # In probability space: p_{next}(y) = sum_x p_t(x) * (delta(x,y) + h * R_bar(x,y))
        # = p_t(y) + h * sigma(t)/K * sum_{x != y} p_t(x) * score(y|x)

        # For the soft version, score_output represents the expected score
        transition = h * sigma_t / K * score_output  # [B, L, K]

        # Outgoing flow from each state
        outgoing = transition.sum(dim=-1, keepdim=True) * p_t.sum(dim=-1, keepdim=True)

        p_next = p_t + transition - outgoing * p_t / (p_t.sum(dim=-1, keepdim=True) + 1e-10)

        # Normalize
        p_next = torch.clamp(p_next, min=0.0)
        p_next = p_next / (p_next.sum(dim=-1, keepdim=True) + 1e-10)

        return p_next
