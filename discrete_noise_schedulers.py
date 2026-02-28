"""
Discrete Noise Schedulers for CTMC-based Discrete Diffusion Models.

Implements forward processes for:
  - Absorbing (Mask) Diffusion: tokens transition to [MASK] state
  - Uniform Diffusion: tokens transition uniformly to any state
  - Discrete Flow Matching: interpolation between source and target distributions

Each scheduler provides:
  - rate_matrix(t): the CTMC rate matrix R_t at time t
  - transition_matrix(t, s): the transition matrix Q_{t|s} from time s to t
  - marginal_prob(t): the marginal corruption probability beta_t
  - rate_scalar(t): the scalar rate sigma(t)
  - inverse_rate_integral(v): the inverse of integral_0^t sigma(s) ds = v
"""

import torch
import math
from abc import ABC, abstractmethod


class DiscreteNoiseSchedule(ABC):
    """Base class for discrete diffusion noise schedules (CTMC formulation)."""

    def __init__(self, vocab_size, T=1.0, eps=1e-4):
        """
        Args:
            vocab_size: Size of the discrete state space K.
            T: Terminal time of the forward process.
            eps: Small epsilon for numerical stability / early stopping.
        """
        self.vocab_size = vocab_size
        self.K = vocab_size
        self.T = T
        self.eps = eps

    @abstractmethod
    def rate_scalar(self, t):
        """Compute sigma(t), the scalar rate function at time t.

        The rate matrix is R_t = sigma(t) * R_base.

        Args:
            t: Time in [0, T]. Can be a scalar or tensor.
        Returns:
            sigma(t) as a tensor.
        """
        pass

    @abstractmethod
    def rate_integral(self, t):
        """Compute integral_0^t sigma(s) ds.

        Used for computing transition probabilities in closed form.
        """
        pass

    @abstractmethod
    def inverse_rate_integral(self, v):
        """Compute t such that integral_0^t sigma(s) ds = v.

        Used for schedule parameterization in rate-integral space.
        """
        pass

    @abstractmethod
    def marginal_prob(self, t):
        """Compute beta_t = 1 - exp(-integral_0^t sigma(s) ds).

        This is the probability that a token has been corrupted by time t.

        Returns:
            beta_t as a tensor.
        """
        pass

    @abstractmethod
    def transition_matrix(self, t, s=0.0):
        """Compute the transition matrix Q_{t|s} from time s to time t.

        Args:
            t: Target time.
            s: Source time (default 0).
        Returns:
            Q_{t|s} of shape [K, K].
        """
        pass

    @abstractmethod
    def rate_matrix(self, t):
        """Compute the rate matrix R_t at time t.

        Returns:
            R_t of shape [K, K]. Rows sum to zero.
        """
        pass

    def marginal_lambda(self, t):
        """Compute lambda_t = log(1 - beta_t) = -integral_0^t sigma(s) ds.

        Analogous to log-SNR for continuous diffusion.
        This is a monotonically decreasing function of t.
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return -self.rate_integral(t)

    def inverse_lambda(self, lamb):
        """Compute t from lambda_t.

        lamb = -integral_0^t sigma(s) ds, so t = inverse_rate_integral(-lamb).
        """
        if not isinstance(lamb, torch.Tensor):
            lamb = torch.tensor(lamb, dtype=torch.float64)
        return self.inverse_rate_integral(-lamb)

    @property
    def lambda_max(self):
        """Lambda at t=eps (least corrupted). Most negative rate integral."""
        return self.marginal_lambda(self.eps).item()

    @property
    def lambda_min(self):
        """Lambda at t=T (most corrupted). Most negative."""
        return self.marginal_lambda(self.T).item()


class AbsorbingSchedule(DiscreteNoiseSchedule):
    """Absorbing (Mask) diffusion schedule.

    Forward process: each token independently transitions to [MASK] state
    at rate sigma(t). The mask state is indexed as K-1 (last token).

    Supports multiple rate parameterizations:
      - 'linear': sigma(t) = beta_0 + t * (beta_1 - beta_0)
      - 'geometric': sigma(t) = sigma_min * (sigma_max/sigma_min)^t
      - 'log_linear': sigma(t) = 1 / (1 - t + eps), giving beta_t = t / (1 + eps)
      - 'cosine': sigma(t) derived from cosine schedule on beta_t

    Reference: Austin et al. (D3PM, 2021), Sahoo et al. (MDLM, 2024)
    """

    def __init__(
        self,
        vocab_size,
        schedule='log_linear',
        T=1.0,
        eps=1e-4,
        beta_0=0.1,
        beta_1=20.0,
    ):
        super().__init__(vocab_size, T, eps)
        self.schedule = schedule
        self.mask_index = vocab_size - 1  # [MASK] is the last token
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def rate_scalar(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)

        if self.schedule == 'log_linear':
            # sigma(t) = 1 / (1 - t + eps), so integral = -log(1 - t + eps) + log(1 + eps)
            return 1.0 / (1.0 - t + self.eps)
        elif self.schedule == 'linear':
            return self.beta_0 + t * (self.beta_1 - self.beta_0)
        elif self.schedule == 'cosine':
            # Derived from beta_t = 1 - cos(pi*t/2 / (1+s))^2 / cos(pi*s/2/(1+s))^2
            s = 0.008
            f_t = torch.cos((t / self.T + s) / (1 + s) * math.pi / 2)
            f_0 = math.cos(s / (1 + s) * math.pi / 2)
            # beta_t = 1 - (f_t / f_0)^2
            # sigma(t) = -d/dt log(1 - beta_t) = -2 * f_t' / f_t
            f_t_prime = -(math.pi / (2 * self.T * (1 + s))) * torch.sin(
                (t / self.T + s) / (1 + s) * math.pi / 2
            )
            return -2.0 * f_t_prime / f_t
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def rate_integral(self, t):
        """integral_0^t sigma(s) ds"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)

        if self.schedule == 'log_linear':
            return -torch.log(1.0 - t + self.eps) + math.log(1.0 + self.eps)
        elif self.schedule == 'linear':
            return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t ** 2
        elif self.schedule == 'cosine':
            s = 0.008
            f_t = torch.cos((t / self.T + s) / (1 + s) * math.pi / 2)
            f_0 = math.cos(s / (1 + s) * math.pi / 2)
            return -2.0 * torch.log(f_t / f_0)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def inverse_rate_integral(self, v):
        """Find t such that integral_0^t sigma(s) ds = v."""
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float64)

        if self.schedule == 'log_linear':
            # v = -log(1 - t + eps) + log(1 + eps)
            # 1 - t + eps = (1 + eps) * exp(-v)
            # t = 1 + eps - (1 + eps) * exp(-v)
            return 1.0 + self.eps - (1.0 + self.eps) * torch.exp(-v)
        elif self.schedule == 'cosine':
            s = 0.008
            f_0 = math.cos(s / (1 + s) * math.pi / 2)
            # v = -2 * log(f_t / f_0) => f_t = f_0 * exp(-v/2)
            f_t = f_0 * torch.exp(-v / 2)
            # f_t = cos((t/T + s) / (1+s) * pi/2)
            # arccos(f_t) = (t/T + s) / (1+s) * pi/2
            arg = torch.clamp(f_t, -1.0, 1.0)
            t = (torch.arccos(arg) * 2 * (1 + s) / math.pi - s) * self.T
            return t
        else:
            # For 'linear', use numerical inversion via bisection
            return self._numerical_inverse_rate_integral(v)

    def _numerical_inverse_rate_integral(self, v, num_iters=64):
        """Bisection-based numerical inversion."""
        lo = torch.zeros_like(v)
        hi = torch.full_like(v, self.T)
        for _ in range(num_iters):
            mid = (lo + hi) / 2
            val = self.rate_integral(mid)
            lo = torch.where(val < v, mid, lo)
            hi = torch.where(val >= v, mid, hi)
        return (lo + hi) / 2

    def marginal_prob(self, t):
        """beta_t = 1 - exp(-integral_0^t sigma(s) ds)"""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return 1.0 - torch.exp(-self.rate_integral(t))

    def transition_matrix(self, t, s=0.0):
        """Q_{t|s} for absorbing diffusion.

        Q_{t|s}(i, i) = exp(-(integral_s^t sigma(u) du))       for i != MASK
        Q_{t|s}(i, MASK) = 1 - exp(-(integral_s^t sigma(u) du)) for i != MASK
        Q_{t|s}(MASK, MASK) = 1

        Returns:
            Q of shape [K, K].
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float64)

        integral = self.rate_integral(t) - self.rate_integral(s)
        survival = torch.exp(-integral)  # prob of NOT being absorbed

        K = self.K
        Q = torch.zeros(K, K, dtype=torch.float64)
        # Non-mask states: survive with prob survival, absorb with prob 1-survival
        for i in range(K - 1):
            Q[i, i] = survival
            Q[i, self.mask_index] = 1.0 - survival
        # Mask state: always stays
        Q[self.mask_index, self.mask_index] = 1.0

        return Q

    def rate_matrix(self, t):
        """R_t = sigma(t) * R_base for absorbing diffusion.

        R_base(i, MASK) = 1 for i != MASK
        R_base(i, i) = -1 for i != MASK
        R_base(MASK, :) = 0

        Returns:
            R_t of shape [K, K].
        """
        sigma_t = self.rate_scalar(t)
        K = self.K
        R = torch.zeros(K, K, dtype=torch.float64)
        for i in range(K - 1):
            R[i, self.mask_index] = sigma_t
            R[i, i] = -sigma_t
        return R

    def rate_matrix_batch(self, t, device=None):
        """Efficient batched rate matrix computation.

        For absorbing diffusion, R_t is sparse: only 2 non-zero entries per row
        (diagonal and mask column). We return the scalar rate for efficiency.

        Args:
            t: Tensor of times, shape [B] or scalar.
        Returns:
            sigma_t: Rate scalars, shape [B] or scalar.
        """
        return self.rate_scalar(t)


class UniformSchedule(DiscreteNoiseSchedule):
    """Uniform diffusion schedule.

    Forward process: each token independently transitions to a uniformly
    random state at rate sigma(t).

    R_base = (1/K) * 1*1^T - I

    Transition matrix: Q_{t|0}(i, j) = (1 - beta_t) * delta(i,j) + beta_t / K

    Reference: Austin et al. (D3PM, 2021), Lou et al. (SEDD, 2024)
    """

    def __init__(
        self,
        vocab_size,
        schedule='log_linear',
        T=1.0,
        eps=1e-4,
        beta_0=0.1,
        beta_1=20.0,
    ):
        super().__init__(vocab_size, T, eps)
        self.schedule = schedule
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def rate_scalar(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)

        if self.schedule == 'log_linear':
            return 1.0 / (1.0 - t + self.eps)
        elif self.schedule == 'linear':
            return self.beta_0 + t * (self.beta_1 - self.beta_0)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def rate_integral(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)

        if self.schedule == 'log_linear':
            return -torch.log(1.0 - t + self.eps) + math.log(1.0 + self.eps)
        elif self.schedule == 'linear':
            return self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t ** 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def inverse_rate_integral(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float64)

        if self.schedule == 'log_linear':
            return 1.0 + self.eps - (1.0 + self.eps) * torch.exp(-v)
        else:
            # Numerical inversion
            lo = torch.zeros_like(v)
            hi = torch.full_like(v, self.T)
            for _ in range(64):
                mid = (lo + hi) / 2
                val = self.rate_integral(mid)
                lo = torch.where(val < v, mid, lo)
                hi = torch.where(val >= v, mid, hi)
            return (lo + hi) / 2

    def marginal_prob(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        return 1.0 - torch.exp(-self.rate_integral(t))

    def transition_matrix(self, t, s=0.0):
        """Q_{t|s} for uniform diffusion.

        Q_{t|s}(i, j) = (1 - beta) * delta(i,j) + beta / K

        where beta = 1 - exp(-(integral_s^t sigma(u) du)).
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float64)

        integral = self.rate_integral(t) - self.rate_integral(s)
        survival = torch.exp(-integral)
        beta = 1.0 - survival

        K = self.K
        Q = survival * torch.eye(K, dtype=torch.float64) + (beta / K) * torch.ones(K, K, dtype=torch.float64)
        return Q

    def rate_matrix(self, t):
        """R_t = sigma(t) * ((1/K) * 1*1^T - I)"""
        sigma_t = self.rate_scalar(t)
        K = self.K
        R = sigma_t * ((1.0 / K) * torch.ones(K, K, dtype=torch.float64) - torch.eye(K, dtype=torch.float64))
        return R


class DiscreteFlowSchedule(DiscreteNoiseSchedule):
    """Discrete Flow Matching schedule.

    The probability path interpolates between source and target:
        p_{t|0,1}(x | x_0, x_1) = (1 - kappa(t)) * delta(x, x_0) + kappa(t) * delta(x, x_1)

    where kappa(t) is the interpolation schedule.

    The corresponding rate matrix in the CTMC formulation:
        R_t(x, y) = kappa'(t) / (1 - kappa(t)) * (delta(y, x_1) - delta(x, y))  [for x != x_1]

    Reference: Gat et al. (Discrete Flow Matching, NeurIPS 2024)
    """

    def __init__(
        self,
        vocab_size,
        schedule='linear',
        T=1.0,
        eps=1e-4,
    ):
        super().__init__(vocab_size, T, eps)
        self.schedule = schedule

    def kappa(self, t):
        """Interpolation function kappa(t) in [0, 1]."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        t_normalized = t / self.T

        if self.schedule == 'linear':
            return t_normalized
        elif self.schedule == 'cosine':
            return 1.0 - torch.cos(t_normalized * math.pi / 2)
        elif self.schedule == 'quadratic':
            return t_normalized ** 2
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def kappa_prime(self, t):
        """Derivative of kappa(t)."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        t_normalized = t / self.T

        if self.schedule == 'linear':
            return torch.ones_like(t) / self.T
        elif self.schedule == 'cosine':
            return (math.pi / (2 * self.T)) * torch.sin(t_normalized * math.pi / 2)
        elif self.schedule == 'quadratic':
            return 2 * t_normalized / self.T
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def rate_scalar(self, t):
        """Effective rate: kappa'(t) / (1 - kappa(t))."""
        k = self.kappa(t)
        kp = self.kappa_prime(t)
        return kp / (1.0 - k + self.eps)

    def rate_integral(self, t):
        """integral_0^t kappa'(s) / (1 - kappa(s)) ds = -log(1 - kappa(t))."""
        k = self.kappa(t)
        return -torch.log(1.0 - k + self.eps)

    def inverse_rate_integral(self, v):
        """Find t such that -log(1 - kappa(t)) = v, i.e., kappa(t) = 1 - exp(-v)."""
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float64)
        target_kappa = 1.0 - torch.exp(-v)
        # Invert kappa
        if self.schedule == 'linear':
            return target_kappa * self.T
        elif self.schedule == 'cosine':
            return torch.arccos(1.0 - target_kappa) * 2 * self.T / math.pi
        elif self.schedule == 'quadratic':
            return torch.sqrt(target_kappa) * self.T
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def marginal_prob(self, t):
        return self.kappa(t)

    def transition_matrix(self, t, s=0.0):
        """For DFM, the transition depends on the target x_1.

        This returns the marginal transition matrix under uniform x_1 assumption,
        which is equivalent to the uniform diffusion transition.
        For conditional generation, use the score-based sampler directly.
        """
        k_t = self.kappa(t)
        k_s = self.kappa(s) if s > 0 else torch.tensor(0.0, dtype=torch.float64)

        # Conditional survival: P(still at source | was at source at time s)
        if isinstance(k_s, (int, float)):
            k_s = torch.tensor(k_s, dtype=torch.float64)
        survival = (1.0 - k_t) / (1.0 - k_s + self.eps)

        K = self.K
        Q = survival * torch.eye(K, dtype=torch.float64) + (1.0 - survival) / K * torch.ones(K, K, dtype=torch.float64)
        return Q

    def rate_matrix(self, t):
        """Marginal rate matrix (under uniform target assumption)."""
        sigma_t = self.rate_scalar(t)
        K = self.K
        R = sigma_t * ((1.0 / K) * torch.ones(K, K, dtype=torch.float64) - torch.eye(K, dtype=torch.float64))
        return R
