"""
Probability Flow solver for differentiable schedule optimization in LD3-D.

Instead of tracking discrete samples, this solver operates on full probability
distributions per position. This enables gradient-based optimization of the
timestep schedule, since all operations are differentiable.

The key insight: for schedule optimization, we don't need discrete samples.
We track p_{t_i}^l in Delta^K for each position l, and the loss is computed
on the final distributions p_0^l vs. teacher distributions.

This is the core differentiable component that enables LD3-D training.
"""

import torch
import torch.nn.functional as F

from .discrete_solver_base import DiscreteSolver


class ProbabilityFlowSolver(DiscreteSolver):
    """Differentiable probability-flow solver for LD3-D schedule optimization.

    Operates on [B, L, K] probability tensors instead of [B, L] token tensors.
    All operations are differentiable w.r.t. timestep parameters.
    """

    def __init__(self, noise_schedule, diffusion_type='absorbing', temperature=1.0):
        """
        Args:
            noise_schedule: DiscreteNoiseSchedule instance
            diffusion_type: 'absorbing' or 'uniform'
            temperature: Temperature for softmax in score computation
        """
        super().__init__(noise_schedule, diffusion_type)
        self.temperature = temperature

    def step(self, x_t, t_cur, t_next, score_fn, **kwargs):
        """Standard discrete sampling step (non-differentiable).

        For actual generation, delegates to tau-leaping logic.
        """
        from .tau_leaping import TauLeapingSolver
        tau_solver = TauLeapingSolver(self.noise_schedule, self.diffusion_type)
        return tau_solver.step(x_t, t_cur, t_next, score_fn, **kwargs)

    def sample_probability_flow(self, score_model, p_init, timesteps, timesteps2=None,
                                condition=None, return_intermediates=False):
        """Run the full reverse process in probability space.

        This is the main method used during LD3-D training.

        Args:
            score_model: The score network. Takes soft token embeddings + time -> logits.
                Interface: score_model(p_t, t, condition=None) -> [B, L, K]
                where p_t is [B, L, K] probabilities (used as soft embeddings).
            p_init: Initial distribution at t=T, shape [B, L, K].
                For absorbing: one-hot on [MASK] for all positions.
                For uniform: uniform 1/K distribution.
            timesteps: Primary schedule, shape [N+1], decreasing.
            timesteps2: Perturbed schedule for score evaluation, shape [N+1].
            condition: Optional conditioning tensor.
            return_intermediates: If True, return list of all intermediate distributions.

        Returns:
            p_final: Final distribution at t=0, shape [B, L, K].
            intermediates: (optional) List of [B, L, K] distributions at each step.
        """
        if timesteps2 is None:
            timesteps2 = timesteps

        N = len(timesteps) - 1
        p_cur = p_init
        intermediates = [p_cur] if return_intermediates else None

        for i in range(N):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            t_eval = timesteps2[i]  # Perturbed time for score evaluation

            # Step size (differentiable w.r.t. timesteps!)
            h = (t_cur - t_next).abs()

            # Score model evaluation with soft input
            score_output = self._evaluate_score(score_model, p_cur, t_eval, condition)

            # Update probability distribution
            if self.diffusion_type == 'absorbing':
                p_cur = self._prob_step_absorbing(p_cur, h, t_cur, score_output)
            elif self.diffusion_type == 'uniform':
                p_cur = self._prob_step_uniform(p_cur, h, t_cur, score_output)
            else:
                p_cur = self._prob_step_uniform(p_cur, h, t_cur, score_output)

            if return_intermediates:
                intermediates.append(p_cur)

        if return_intermediates:
            return p_cur, intermediates
        return p_cur

    def _evaluate_score(self, score_model, p_t, t, condition):
        """Evaluate score model on soft probability input.

        The score model typically expects token indices. For probability-flow,
        we pass the probability distribution and the model uses a soft embedding:
            emb = p_t @ embedding_matrix  (weighted average of embeddings)

        Args:
            score_model: The neural network
            p_t: [B, L, K] probability distributions
            t: Time (scalar or tensor)
            condition: Optional conditioning

        Returns:
            logits: [B, L, K] unnormalized log-probabilities
        """
        if condition is not None:
            return score_model(p_t, t, condition=condition)
        return score_model(p_t, t)

    def _prob_step_absorbing(self, p_t, h, t_cur, score_output):
        """Differentiable probability update for absorbing diffusion.

        The exact update under the reverse CTMC:
            dp/dt = p * R_bar_t

        For absorbing diffusion, R_bar_t transfers mass from MASK to clean tokens:
            R_bar(MASK, y) = sigma(t) * p_theta(x_0 = y | x_t)  for y != MASK
            R_bar(x, x) = -sigma(t) * delta(x, MASK)             for x = MASK

        In probability space (first-order Euler / tau-leaping):
            p_{t-h}(y) = p_t(y) + h * sigma(t) * p_t(MASK) * p_theta(y)  for y != MASK
            p_{t-h}(MASK) = p_t(MASK) * (1 - h * sigma(t))

        Or using the exact matrix exponential:
            p_{t-h}(y) = p_t(y) + (1 - exp(-sigma*h)) * p_t(MASK) * p_theta(y)
            p_{t-h}(MASK) = p_t(MASK) * exp(-sigma*h)
        """
        mask_idx = self.noise_schedule.mask_index
        K = self.K
        device = p_t.device

        # Score -> denoising distribution
        p_theta = F.softmax(score_output / self.temperature, dim=-1)  # [B, L, K]

        # Zero out mask token in prediction
        p_theta_clean = p_theta.clone()
        p_theta_clean[:, :, mask_idx] = 0.0
        # Renormalize
        denom = p_theta_clean.sum(dim=-1, keepdim=True) + 1e-10
        p_theta_clean = p_theta_clean / denom

        # Rate
        sigma_t = self.noise_schedule.rate_scalar(t_cur)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t = sigma_t.to(device)

        # Exact (matrix exponential) update
        # Use exact formulation: survival = exp(-sigma*h)
        survival = torch.exp(-sigma_t * h)
        transfer_rate = 1.0 - survival  # Fraction of mask mass to transfer

        # Current mask probability
        p_mask = p_t[:, :, mask_idx:mask_idx + 1]  # [B, L, 1]

        # Flow from MASK to clean tokens
        flow = transfer_rate * p_mask * p_theta_clean  # [B, L, K]

        # Update
        p_next = p_t.clone()
        p_next = p_next + flow
        p_next[:, :, mask_idx:mask_idx + 1] = p_mask * survival

        # Ensure valid distribution
        p_next = torch.clamp(p_next, min=1e-10)
        p_next = p_next / p_next.sum(dim=-1, keepdim=True)

        return p_next

    def _prob_step_uniform(self, p_t, h, t_cur, score_output):
        """Differentiable probability update for uniform diffusion.

        Under the reverse CTMC with concrete score:
            R_bar(x, y) = sigma(t)/K * s_theta(y|x) for y != x

        In probability space, the update becomes:
            p_{t-h}(y) = sum_x p_t(x) * [delta(x,y) + h * R_bar(x,y)]

        For the differentiable version, we compute:
            p_{t-h} = p_t + h * sigma(t)/K * [score_weighted_mass - outflow]

        The exact version uses:
            p_{t-h} = (1 - alpha) * p_t + alpha * p_denoised
        where alpha = 1 - exp(-sigma*h) and p_denoised = softmax(score_output).
        """
        K = self.K
        device = p_t.device

        sigma_t = self.noise_schedule.rate_scalar(t_cur)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t = sigma_t.to(device)

        # Denoised distribution from score
        p_denoised = F.softmax(score_output / self.temperature, dim=-1)  # [B, L, K]

        # Exact mixing coefficient
        alpha = 1.0 - torch.exp(-sigma_t * h)
        alpha = torch.clamp(alpha, 0.0, 1.0 - 1e-6)

        # Exponential mixing
        p_next = (1.0 - alpha) * p_t + alpha * p_denoised

        # Ensure valid distribution
        p_next = torch.clamp(p_next, min=1e-10)
        p_next = p_next / p_next.sum(dim=-1, keepdim=True)

        return p_next

    def compute_loss(self, p_final, targets, loss_type='cross_entropy'):
        """Compute loss between final probability distribution and target tokens.

        Args:
            p_final: Predicted distributions, shape [B, L, K]
            targets: Target token indices, shape [B, L]
            loss_type: 'cross_entropy', 'kl', or 'l2'

        Returns:
            loss: Scalar loss value (differentiable)
        """
        B, L, K = p_final.shape

        if loss_type == 'cross_entropy':
            # Standard cross-entropy: -log p_final(target)
            log_probs = torch.log(p_final + 1e-10)
            target_log_probs = log_probs.gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)  # [B, L]
            loss = -target_log_probs.mean()

        elif loss_type == 'kl':
            # KL divergence from one-hot target to predicted
            target_one_hot = F.one_hot(targets, K).float()
            log_probs = torch.log(p_final + 1e-10)
            kl = (target_one_hot * (torch.log(target_one_hot + 1e-10) - log_probs)).sum(dim=-1)
            loss = kl.mean()

        elif loss_type == 'l2':
            # L2 distance between predicted probs and one-hot target
            target_one_hot = F.one_hot(targets, K).float()
            loss = ((p_final - target_one_hot) ** 2).sum(dim=-1).mean()

        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        return loss

    def compute_theory_proxy_loss(self, score_model, timesteps, condition=None):
        """Compute the theory-motivated proxy loss (Section 4.3c of proposal).

        L_proxy = sum_i h_i^2 * C_hat_i

        where C_hat_i estimates the rate-of-change of the reverse rate matrix
        at timestep t_i.

        This loss can be used as a regularizer or standalone objective.

        Args:
            score_model: Score network
            timesteps: Current schedule, shape [N+1]
            condition: Optional conditioning

        Returns:
            loss_proxy: Scalar loss value
        """
        N = len(timesteps) - 1
        loss = torch.tensor(0.0, device=timesteps.device, requires_grad=True)

        for i in range(N):
            h_i = (timesteps[i] - timesteps[i + 1]).abs()

            # Estimate C_i via finite difference of rate
            t_i = timesteps[i]
            dt = 1e-3
            sigma_i = self.noise_schedule.rate_scalar(t_i)
            sigma_i_plus = self.noise_schedule.rate_scalar(t_i + dt)

            # C_i ~ |d(sigma)/dt|^2
            dsigma_dt = ((sigma_i_plus - sigma_i) / dt) ** 2
            if isinstance(dsigma_dt, torch.Tensor):
                dsigma_dt = dsigma_dt.to(timesteps.device)

            loss = loss + h_i ** 2 * dsigma_dt

        return loss
