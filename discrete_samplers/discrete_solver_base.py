"""
Base class for discrete diffusion solvers.

All discrete solvers implement the reverse process of a CTMC-based
discrete diffusion model, taking a sequence of corrupted tokens and
producing cleaner tokens step by step.
"""

import torch
from abc import ABC, abstractmethod


class DiscreteSolver(ABC):
    """Base class for discrete diffusion samplers.

    All solvers share the same interface:
      - sample_simple: given noise tokens and a timestep schedule, produce clean tokens
      - step: perform a single reverse step from t_i to t_{i-1}

    The solver uses a score model s_theta(x_t, t) that estimates the concrete score
    (ratio p_t(y)/p_t(x)) for each token position.
    """

    def __init__(self, noise_schedule, diffusion_type='absorbing'):
        """
        Args:
            noise_schedule: A DiscreteNoiseSchedule instance (Absorbing, Uniform, etc.)
            diffusion_type: 'absorbing' or 'uniform' or 'flow'
        """
        self.noise_schedule = noise_schedule
        self.diffusion_type = diffusion_type
        self.K = noise_schedule.vocab_size

    @abstractmethod
    def step(self, x_t, t_cur, t_next, score_fn, **kwargs):
        """Perform one reverse step: x_{t_cur} -> x_{t_next}.

        Args:
            x_t: Current tokens, shape [B, L] (long tensor of token indices)
            t_cur: Current time (scalar or tensor)
            t_next: Next time (t_next < t_cur, moving toward clean data)
            score_fn: Function (x, t) -> score tensor.
                For absorbing: returns [B, L, K] log-probabilities of clean token given masked position.
                For uniform/SEDD: returns [B, L, K] concrete score ratios p_t(y)/p_t(x).
            **kwargs: Solver-specific parameters.

        Returns:
            x_next: Updated tokens at time t_next, shape [B, L]
        """
        pass

    def sample_simple(self, model_fn, x, timesteps, timesteps2=None,
                      order=1, NFEs=None, condition=None,
                      unconditional_condition=None, **kwargs):
        """Sample from the reverse process using the given timestep schedule.

        Interface compatible with LD3's ODE solvers.

        Args:
            model_fn: Score model function (x, t, [condition]) -> scores
            x: Initial noisy tokens, shape [B, L]
            timesteps: Primary timestep schedule, shape [N+1], decreasing from T to 0
            timesteps2: Perturbed timestep schedule (for LD3-D), shape [N+1]
            order: Solver order (1 for tau-leaping, 2 for theta-trapezoidal)
            NFEs: Number of function evaluations (= len(timesteps) - 1)
            condition: Optional conditioning information
            unconditional_condition: Optional unconditional conditioning (for guidance)
            **kwargs: Additional solver parameters

        Returns:
            x_0: Generated tokens, shape [B, L]
        """
        if timesteps2 is None:
            timesteps2 = timesteps

        N = len(timesteps) - 1
        x_cur = x.clone()

        for i in range(N):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            # Use timesteps2 for score evaluation time (perturbation)
            t_eval = timesteps2[i]

            def score_fn(x_in, t_in):
                if condition is not None:
                    return model_fn(x_in, t_in, condition)
                return model_fn(x_in, t_in)

            x_cur = self.step(x_cur, t_cur, t_next, score_fn, t_eval=t_eval, **kwargs)

        return x_cur

    def sample_probability_flow(self, model_fn, p_init, timesteps, timesteps2=None,
                                condition=None, **kwargs):
        """Sample in probability space (for differentiable schedule optimization).

        Instead of tracking discrete tokens, track the full probability distribution
        per position. This is differentiable w.r.t. the timestep schedule.

        Args:
            model_fn: Score model that accepts soft inputs
            p_init: Initial probability distribution, shape [B, L, K]
            timesteps: Timestep schedule
            timesteps2: Perturbed schedule
            condition: Optional conditioning

        Returns:
            p_final: Final probability distribution, shape [B, L, K]
        """
        raise NotImplementedError("Subclasses should implement probability_flow for differentiable optimization")

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute baseline timestep schedules for discrete diffusion.

        Args:
            skip_type: Schedule type ('uniform', 'rate_uniform', 'quadratic', 'cosine')
            t_T: Start time (maximum corruption)
            t_0: End time (minimum corruption)
            N: Number of steps
            device: torch device

        Returns:
            timesteps: Tensor of shape [N+1], decreasing from t_T to t_0
        """
        if skip_type == 'uniform':
            # Uniform in time
            return torch.linspace(t_T, t_0, N + 1).to(device)

        elif skip_type == 'rate_uniform':
            # Uniform in rate-integral space (analogous to logSNR-uniform)
            v_T = self.noise_schedule.rate_integral(torch.tensor(t_T)).item()
            v_0 = self.noise_schedule.rate_integral(torch.tensor(t_0)).item()
            v_steps = torch.linspace(v_T, v_0, N + 1)
            return self.noise_schedule.inverse_rate_integral(v_steps).float().to(device)

        elif skip_type == 'quadratic':
            # Quadratic in time
            rho = 2.0
            ramp = torch.linspace(0, 1, N + 1)
            t_steps = t_0 + (t_T - t_0) * (1 - ramp) ** rho
            return t_steps.to(device)

        elif skip_type == 'cosine':
            # Cosine schedule
            ramp = torch.linspace(0, 1, N + 1)
            t_steps = t_0 + (t_T - t_0) * 0.5 * (1 + torch.cos(ramp * torch.pi))
            return t_steps.to(device)

        else:
            raise ValueError(f"Unknown skip_type: {skip_type}")

    def prepare_timesteps(self, steps=None, t_start=None, t_end=None,
                          skip_type='uniform', device=None, load_from=None):
        """Prepare timestep schedules (learned or baseline)."""
        import os
        if load_from is not None and os.path.isfile(load_from):
            return self.prepare_learn_timesteps(load_from=load_from, device=device)

        timesteps = self.get_time_steps(
            skip_type=skip_type, t_T=t_start, t_0=t_end, N=steps, device=device
        )
        return timesteps, timesteps

    def prepare_learn_timesteps(self, load_from, device=None):
        """Load learned timesteps from checkpoint."""
        checkpoint = torch.load(load_from, map_location=device)
        timesteps = checkpoint['best_t_steps']
        length = timesteps.shape[0] // 2
        return timesteps[:length].to(device), timesteps[length:].to(device)
