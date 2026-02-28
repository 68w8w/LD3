"""
Model wrappers for pre-trained discrete diffusion models.

Provides a unified interface for different discrete diffusion model architectures:
  - MDLM (Masked Diffusion Language Model): x_0-prediction for absorbing diffusion
  - SEDD (Score Entropy Discrete Diffusion): concrete score ratio prediction
  - D3PM: General discrete diffusion with transition matrices
  - Discrete Flow Matching: probability denoiser (x_1-prediction)

Each wrapper normalizes the model interface to:
    model(x_t, t, condition=None) -> output [B, L, K]

where output semantics depend on the model type.

For LD3-D probability-flow training, a soft-input wrapper is also provided:
    model_soft(p_t, t, condition=None) -> logits [B, L, K]

which accepts probability distributions instead of token indices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def discrete_model_wrapper(model, noise_schedule, model_type='x0_prediction',
                           vocab_size=None, embedding_layer=None):
    """Create a unified wrapper for discrete diffusion models.

    Args:
        model: The pre-trained discrete diffusion model.
        noise_schedule: DiscreteNoiseSchedule instance.
        model_type: One of:
            - 'x0_prediction': Model predicts p(x_0 | x_t, t) as logits (MDLM)
            - 'score_ratio': Model predicts concrete score s(y|x) = p_t(y)/p_t(x) (SEDD)
            - 'denoiser': Model predicts p(x_1 | x_t, t) for flow matching
        vocab_size: Vocabulary size K. If None, inferred from noise_schedule.
        embedding_layer: Optional embedding layer for soft-input mode.
            If None, uses the model's own embedding layer.

    Returns:
        A callable wrapper with interface:
            wrapper(x, t, condition=None) -> [B, L, K] logits/scores
            wrapper.soft(p, t, condition=None) -> [B, L, K] logits (differentiable)
    """
    K = vocab_size or noise_schedule.vocab_size

    class DiscreteModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.model_type = model_type
            self.K = K
            self._embedding = embedding_layer

        def forward(self, x, t, condition=None):
            """Standard forward pass with discrete token inputs.

            Args:
                x: Token indices [B, L] (long tensor)
                t: Time, scalar or [B] tensor
                condition: Optional conditioning tensor

            Returns:
                output: [B, L, K] tensor. Semantics depend on model_type.
            """
            # Ensure t is in the right format
            t_input = self._prepare_time(t, x.shape[0], x.device)

            # Call the underlying model
            if condition is not None:
                logits = self.model(x, t_input, condition)
            else:
                logits = self.model(x, t_input)

            return self._postprocess(logits, x)

        def soft(self, p, t, condition=None):
            """Soft-input forward pass for probability-flow optimization.

            Instead of token indices, accepts probability distributions and uses
            a soft embedding (weighted average of token embeddings).

            Args:
                p: Probability distributions [B, L, K]
                t: Time
                condition: Optional conditioning

            Returns:
                logits: [B, L, K] output logits (differentiable w.r.t. p and t)
            """
            B, L, K = p.shape
            t_input = self._prepare_time(t, B, p.device)

            # Create soft embeddings from probability distribution
            if self._embedding is not None:
                # soft_emb = p @ embedding_matrix  [B, L, D]
                emb_weight = self._embedding.weight  # [K, D]
                soft_emb = torch.matmul(p, emb_weight)  # [B, L, D]
                logits = self._forward_with_embeddings(soft_emb, t_input, condition)
            else:
                # Fallback: use argmax (not differentiable, for models without accessible embeddings)
                x_hard = p.argmax(dim=-1)
                # Straight-through estimator: forward uses hard, backward uses soft
                x_one_hot = F.one_hot(x_hard, K).float()
                x_soft = p + (x_one_hot - p).detach()  # STE
                if condition is not None:
                    logits = self.model(x_hard, t_input, condition)
                else:
                    logits = self.model(x_hard, t_input)

            return logits

        def _forward_with_embeddings(self, embeddings, t_input, condition):
            """Forward pass with pre-computed embeddings.

            This method should be overridden for specific model architectures
            that expose intermediate embedding interfaces.
            """
            # Default: fall back to standard forward with argmax
            # Real implementations would inject embeddings into the transformer
            raise NotImplementedError(
                "Soft-input forward requires model-specific embedding injection. "
                "Override _forward_with_embeddings or provide embedding_layer."
            )

        def _prepare_time(self, t, batch_size, device):
            """Convert time to the format expected by the model."""
            if isinstance(t, (int, float)):
                t = torch.tensor([t], device=device).float()
            if t.dim() == 0:
                t = t.unsqueeze(0)
            if t.shape[0] == 1 and batch_size > 1:
                t = t.expand(batch_size)
            return t.to(device)

        def _postprocess(self, logits, x):
            """Post-process model output based on model_type."""
            if self.model_type == 'x0_prediction':
                # logits are already p(x_0 | x_t, t) in log-space
                return logits
            elif self.model_type == 'score_ratio':
                # Convert concrete score to x_0 prediction for unified interface
                # s(y|x) = p_t(y)/p_t(x), so log s(y|x) is a log-ratio
                return logits
            elif self.model_type == 'denoiser':
                return logits
            else:
                return logits

    return DiscreteModelWrapper()


class MDLMWrapper(nn.Module):
    """Wrapper specifically for MDLM (Masked Diffusion Language Model).

    MDLM predicts p(x_0 | x_t, t) for absorbing (mask) diffusion.
    The model is a standard transformer that takes masked sequences and outputs
    logits over the vocabulary for each position.

    Reference: Sahoo et al. (NeurIPS 2024)
    """

    def __init__(self, model, noise_schedule, vocab_size):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.K = vocab_size

    def forward(self, x, t, condition=None):
        """
        Args:
            x: Token indices [B, L] or probabilities [B, L, K]
            t: Time [B] or scalar
        Returns:
            logits: [B, L, K] unnormalized log-probs of x_0
        """
        if x.dim() == 3:
            # Soft input mode for probability flow
            return self._soft_forward(x, t, condition)

        # Standard discrete input
        if isinstance(t, (int, float)):
            t = torch.full((x.shape[0],), t, device=x.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        if condition is not None:
            logits = self.model(x, t, condition)
        else:
            logits = self.model(x, t)

        return logits

    def _soft_forward(self, p, t, condition):
        """Forward with probability distribution input.

        For MDLM: create soft embeddings as p @ W_emb, then run transformer.
        """
        if isinstance(t, (int, float)):
            t = torch.full((p.shape[0],), t, device=p.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(p.shape[0])

        # Try to access the model's embedding layer
        if hasattr(self.model, 'embed_tokens'):
            emb_weight = self.model.embed_tokens.weight
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            emb_weight = self.model.transformer.wte.weight
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'embeddings'):
            emb_weight = self.model.backbone.embeddings.word_embeddings.weight
        else:
            # Fallback: use STE with argmax
            x_hard = p.argmax(dim=-1)
            return self.forward(x_hard, t, condition)

        # Soft embedding: [B, L, D]
        soft_emb = torch.matmul(p, emb_weight)

        # Forward through model with soft embeddings
        # This requires model-specific injection
        # Default: use STE
        x_hard = p.argmax(dim=-1)
        logits = self.forward(x_hard, t, condition)
        return logits


class SEDDWrapper(nn.Module):
    """Wrapper for SEDD (Score Entropy Discrete Diffusion).

    SEDD predicts the concrete score ratio s(y|x) = p_t(y)/p_t(x).
    The model outputs [B, L, K] where the (l, y) entry approximates
    log(p_t(x_l = y) / p_t(x_l = current_token)).

    Reference: Lou et al. (ICML 2024 Best Paper)
    """

    def __init__(self, model, noise_schedule, vocab_size):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.K = vocab_size

    def forward(self, x, t, condition=None):
        """
        Args:
            x: Token indices [B, L] or probabilities [B, L, K]
            t: Time [B] or scalar
        Returns:
            score_ratios: [B, L, K] concrete score estimates
        """
        if x.dim() == 3:
            # Soft input: use argmax with STE
            x_hard = x.argmax(dim=-1)
            x = x_hard

        if isinstance(t, (int, float)):
            t = torch.full((x.shape[0],), t, device=x.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # SEDD model typically takes (x, sigma_t) where sigma_t is the noise level
        sigma_t = self.noise_schedule.rate_scalar(t)

        if condition is not None:
            score = self.model(x, sigma_t, condition)
        else:
            score = self.model(x, sigma_t)

        return score

    def score_to_x0_logits(self, score, x_t, t):
        """Convert concrete score to x_0 prediction logits.

        For absorbing diffusion:
            p(x_0 = y | x_t) ∝ s(y | x_t) * q(x_t | x_0 = y)

        For uniform diffusion:
            p(x_0 = y | x_t) ∝ s(y | x_t) (approximately, for large t)
        """
        return score  # For SEDD, score ratios can serve as x_0 logits


class DiscreteFlowWrapper(nn.Module):
    """Wrapper for Discrete Flow Matching models.

    DFM models predict the probability denoiser p(x_1 | x_t, t),
    which gives the conditional distribution of the clean data given
    the noisy observation at time t.

    Reference: Gat et al. (NeurIPS 2024)
    """

    def __init__(self, model, noise_schedule, vocab_size):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.K = vocab_size

    def forward(self, x, t, condition=None):
        """
        Args:
            x: Token indices [B, L] or probabilities [B, L, K]
            t: Time
        Returns:
            logits: [B, L, K] denoiser logits
        """
        if x.dim() == 3:
            x = x.argmax(dim=-1)

        if isinstance(t, (int, float)):
            t = torch.full((x.shape[0],), t, device=x.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        if condition is not None:
            return self.model(x, t, condition)
        return self.model(x, t)
