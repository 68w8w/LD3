"""
Text generation evaluation metrics for LD3-D.

Comprehensive metrics suite for evaluating discrete diffusion text generation:
  - Quality: Gen PPL (computed separately), BLEU vs reference
  - Diversity: Distinct-n, Self-BLEU, Unique token ratio, Entropy
  - Degeneracy detection: Repetition rate, Zipf coefficient
  - Information-theoretic: Bits-per-character (BPC), token entropy

These metrics complement the standard Gen PPL evaluation and provide a
multi-faceted view of generation quality for comparing schedules.

Usage:
    from compute_metrics import compute_all_metrics
    metrics = compute_all_metrics(generated_texts, reference_texts=ref)
"""

import math
import logging
from collections import Counter
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# N-gram utilities
# ============================================================================

def _get_ngrams(tokens, n):
    """Extract n-grams from a token sequence."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _tokenize_char(text):
    """Character-level tokenization (for text8-style data)."""
    return list(text)


def _tokenize_word(text):
    """Simple whitespace tokenization."""
    return text.split()


# ============================================================================
# Diversity Metrics
# ============================================================================

def distinct_n(texts, n, tokenize='char'):
    """Compute Distinct-n: ratio of unique n-grams to total n-grams.

    Higher values indicate more diverse generation.
    Standard metric from Li et al. (2016) "A Diversity-Promoting Objective
    Function for Neural Conversation Models".

    Args:
        texts: List of generated text strings.
        n: N-gram order (1, 2, 3, or 4).
        tokenize: 'char' for character-level, 'word' for word-level.

    Returns:
        Distinct-n score in [0, 1]. Higher = more diverse.
    """
    tok_fn = _tokenize_char if tokenize == 'char' else _tokenize_word
    all_ngrams = []
    for text in texts:
        tokens = tok_fn(text)
        all_ngrams.extend(_get_ngrams(tokens, n))

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def self_bleu(texts, n_max=4, tokenize='char', sample_size=None):
    """Compute Self-BLEU: average BLEU of each sample against all others.

    Lower values indicate more diverse generation across samples.
    From Zhu et al. (2018) "Texygen: A Benchmarking Platform for Text
    Generation Models".

    Args:
        texts: List of generated text strings.
        n_max: Maximum n-gram order for BLEU (typically 4).
        tokenize: 'char' or 'word'.
        sample_size: If set, randomly subsample this many texts for efficiency.

    Returns:
        Self-BLEU score in [0, 1]. Lower = more diverse.
    """
    if len(texts) < 2:
        return 0.0

    tok_fn = _tokenize_char if tokenize == 'char' else _tokenize_word
    tokenized = [tok_fn(t) for t in texts]

    if sample_size and len(tokenized) > sample_size:
        indices = np.random.choice(len(tokenized), sample_size, replace=False)
        tokenized = [tokenized[i] for i in indices]

    bleu_scores = []
    for i in range(len(tokenized)):
        hypothesis = tokenized[i]
        references = [tokenized[j] for j in range(len(tokenized)) if j != i]
        score = _compute_bleu(hypothesis, references, n_max)
        bleu_scores.append(score)

    return float(np.mean(bleu_scores))


def _compute_bleu(hypothesis, references, n_max=4):
    """Compute BLEU score of hypothesis against references (simplified)."""
    if len(hypothesis) == 0:
        return 0.0

    # Collect reference n-gram counts (take max across references for each n-gram)
    precisions = []
    for n in range(1, n_max + 1):
        hyp_ngrams = _get_ngrams(hypothesis, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        hyp_counts = Counter(hyp_ngrams)

        # Max reference counts
        max_ref_counts = Counter()
        for ref in references:
            ref_counts = Counter(_get_ngrams(ref, n))
            for ng, count in ref_counts.items():
                max_ref_counts[ng] = max(max_ref_counts[ng], count)

        # Clipped counts
        clipped = sum(min(hyp_counts[ng], max_ref_counts.get(ng, 0))
                       for ng in hyp_counts)
        total = sum(hyp_counts.values())

        precisions.append(clipped / total if total > 0 else 0.0)

    # Geometric mean of precisions (with smoothing)
    log_prec = 0.0
    for p in precisions:
        if p == 0:
            return 0.0
        log_prec += math.log(p) / len(precisions)

    # Brevity penalty
    closest_ref_len = min(
        (abs(len(ref) - len(hypothesis)), len(ref)) for ref in references
    )[1]
    bp = math.exp(1 - closest_ref_len / len(hypothesis)) if len(hypothesis) < closest_ref_len else 1.0

    return bp * math.exp(log_prec)


def unique_token_ratio(texts, tokenize='char'):
    """Fraction of unique tokens out of all tokens generated.

    Args:
        texts: List of text strings.
        tokenize: 'char' or 'word'.

    Returns:
        Ratio in [0, 1]. Values near 0 suggest degenerate repetition.
    """
    tok_fn = _tokenize_char if tokenize == 'char' else _tokenize_word
    all_tokens = []
    for t in texts:
        all_tokens.extend(tok_fn(t))
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


# ============================================================================
# Information-theoretic Metrics
# ============================================================================

def token_entropy(texts, tokenize='char'):
    """Shannon entropy of the unigram distribution over generated text.

    Higher entropy = more uniform (diverse) token usage.
    Natural language text8 has ~4.17 bits/char entropy.

    Args:
        texts: List of text strings.
        tokenize: 'char' or 'word'.

    Returns:
        Dict with 'unigram_entropy' and 'bigram_entropy' (in bits).
    """
    tok_fn = _tokenize_char if tokenize == 'char' else _tokenize_word
    all_tokens = []
    all_bigrams = []
    for t in texts:
        tokens = tok_fn(t)
        all_tokens.extend(tokens)
        all_bigrams.extend(_get_ngrams(tokens, 2))

    def _entropy(counter):
        total = sum(counter.values())
        if total == 0:
            return 0.0
        probs = np.array(list(counter.values()), dtype=np.float64) / total
        return -np.sum(probs * np.log2(probs + 1e-30))

    return {
        'unigram_entropy': _entropy(Counter(all_tokens)),
        'bigram_entropy': _entropy(Counter(all_bigrams)),
    }


def bits_per_character(gen_ppl):
    """Convert generative perplexity to bits-per-character (BPC).

    Standard metric for text8 evaluation. BPC = log2(PPL).

    Args:
        gen_ppl: Generative perplexity (from GPT-2 evaluation).

    Returns:
        BPC (float). Lower = better.
    """
    if gen_ppl <= 0:
        return float('inf')
    return math.log2(gen_ppl)


# ============================================================================
# Degeneracy Detection
# ============================================================================

def repetition_rate(texts, n=3, tokenize='char'):
    """Intra-sample n-gram repetition rate.

    For each sample, computes fraction of n-grams that appear more than once.
    High repetition suggests mode collapse or degenerate generation.

    Args:
        texts: List of text strings.
        n: N-gram order.
        tokenize: 'char' or 'word'.

    Returns:
        Mean repetition rate across samples, in [0, 1]. Lower = less repetitive.
    """
    tok_fn = _tokenize_char if tokenize == 'char' else _tokenize_word
    rates = []
    for text in texts:
        tokens = tok_fn(text)
        ngrams = _get_ngrams(tokens, n)
        if not ngrams:
            rates.append(0.0)
            continue
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        rates.append(repeated / len(counts))
    return float(np.mean(rates))


def zipf_coefficient(texts, tokenize='char'):
    """Estimate Zipf's law coefficient for generated text.

    Natural language follows Zipf's law: frequency ~ 1/rank^alpha, where
    alpha ~ 1.0. Deviations from ~1.0 indicate unnatural distributions.

    Uses least-squares fit in log-log space.

    Args:
        texts: List of text strings.
        tokenize: 'char' or 'word'.

    Returns:
        Estimated Zipf exponent alpha (float). Natural text ~ 1.0.
    """
    tok_fn = _tokenize_char if tokenize == 'char' else _tokenize_word
    all_tokens = []
    for t in texts:
        all_tokens.extend(tok_fn(t))

    if len(all_tokens) < 10:
        return 0.0

    counts = Counter(all_tokens)
    freqs = sorted(counts.values(), reverse=True)

    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    freqs = np.array(freqs, dtype=np.float64)

    # Least-squares fit: log(freq) = -alpha * log(rank) + c
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs + 1e-30)

    # Simple linear regression
    n = len(log_ranks)
    sum_x = np.sum(log_ranks)
    sum_y = np.sum(log_freqs)
    sum_xy = np.sum(log_ranks * log_freqs)
    sum_x2 = np.sum(log_ranks ** 2)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-30:
        return 0.0

    alpha = -(n * sum_xy - sum_x * sum_y) / denom
    return float(alpha)


# ============================================================================
# Reference-based Metrics
# ============================================================================

def bleu_vs_reference(generated_texts, reference_texts, n_max=4, tokenize='char'):
    """Compute corpus-level BLEU of generated text against reference text.

    Each generated text is scored against ALL reference texts.

    Args:
        generated_texts: List of generated strings.
        reference_texts: List of reference strings.
        n_max: Maximum n-gram order.
        tokenize: 'char' or 'word'.

    Returns:
        Dict with 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4' scores.
    """
    tok_fn = _tokenize_char if tokenize == 'char' else _tokenize_word
    gen_tokenized = [tok_fn(t) for t in generated_texts]
    ref_tokenized = [tok_fn(t) for t in reference_texts]

    if not gen_tokenized or not ref_tokenized:
        return {f'bleu-{n}': 0.0 for n in range(1, n_max + 1)}

    results = {}
    for n in range(1, n_max + 1):
        total_clipped = 0
        total_count = 0
        total_hyp_len = 0
        total_ref_len = 0

        # Build reference n-gram counts
        ref_ngram_counts = Counter()
        for ref in ref_tokenized:
            ref_ngram_counts.update(_get_ngrams(ref, n))

        for hyp in gen_tokenized:
            hyp_ngrams = _get_ngrams(hyp, n)
            if not hyp_ngrams:
                continue
            hyp_counts = Counter(hyp_ngrams)
            for ng, count in hyp_counts.items():
                total_clipped += min(count, ref_ngram_counts.get(ng, 0))
                total_count += count
            total_hyp_len += len(hyp)

        for ref in ref_tokenized:
            total_ref_len += len(ref)
        avg_ref_len = total_ref_len / len(ref_tokenized) if ref_tokenized else 1

        precision = total_clipped / total_count if total_count > 0 else 0.0
        results[f'bleu-{n}'] = precision

    return results


# ============================================================================
# Aggregate
# ============================================================================

def compute_all_metrics(
    generated_texts: List[str],
    reference_texts: Optional[List[str]] = None,
    gen_ppl: Optional[float] = None,
    tokenize: str = 'char',
    self_bleu_samples: int = 100,
) -> Dict[str, float]:
    """Compute all available metrics on generated texts.

    Args:
        generated_texts: List of generated text strings.
        reference_texts: Optional list of reference texts (for BLEU).
        gen_ppl: If provided, also computes BPC from Gen PPL.
        tokenize: 'char' or 'word'.
        self_bleu_samples: Max samples for Self-BLEU (expensive to compute).

    Returns:
        Dict mapping metric_name -> value.
    """
    metrics = {}

    if not generated_texts:
        logger.warning("No generated texts provided.")
        return metrics

    n_samples = len(generated_texts)
    total_chars = sum(len(t) for t in generated_texts)
    metrics['num_samples'] = n_samples
    metrics['avg_length'] = total_chars / n_samples if n_samples > 0 else 0

    # --- Diversity ---
    for n in [1, 2, 3, 4]:
        metrics[f'distinct-{n}'] = distinct_n(generated_texts, n, tokenize)

    metrics['self-bleu'] = self_bleu(
        generated_texts, n_max=4, tokenize=tokenize,
        sample_size=min(self_bleu_samples, n_samples)
    )
    metrics['unique_token_ratio'] = unique_token_ratio(generated_texts, tokenize)

    # --- Information-theoretic ---
    ent = token_entropy(generated_texts, tokenize)
    metrics['unigram_entropy'] = ent['unigram_entropy']
    metrics['bigram_entropy'] = ent['bigram_entropy']

    if gen_ppl is not None:
        metrics['gen_ppl'] = gen_ppl
        metrics['bpc'] = bits_per_character(gen_ppl)

    # --- Degeneracy ---
    metrics['repetition_rate_3gram'] = repetition_rate(generated_texts, n=3, tokenize=tokenize)
    metrics['repetition_rate_5gram'] = repetition_rate(generated_texts, n=5, tokenize=tokenize)
    metrics['zipf_coefficient'] = zipf_coefficient(generated_texts, tokenize)

    # --- Reference-based (if available) ---
    if reference_texts:
        bleu_scores = bleu_vs_reference(
            generated_texts, reference_texts, n_max=4, tokenize=tokenize
        )
        metrics.update(bleu_scores)

    return metrics


def format_metrics_table(metrics: Dict[str, float], title: str = "Generation Metrics") -> str:
    """Format metrics dict as a readable table string.

    Args:
        metrics: Dict from compute_all_metrics.
        title: Table title.

    Returns:
        Formatted string for logging.
    """
    lines = []
    lines.append("=" * 55)
    lines.append(f"{title:^55}")
    lines.append("=" * 55)

    # Group metrics
    groups = {
        'Overview': ['num_samples', 'avg_length'],
        'Quality (lower=better)': ['gen_ppl', 'bpc'],
        'Diversity (higher=better)': [
            'distinct-1', 'distinct-2', 'distinct-3', 'distinct-4',
            'unique_token_ratio',
        ],
        'Diversity (lower=better)': ['self-bleu'],
        'Entropy (bits)': ['unigram_entropy', 'bigram_entropy'],
        'Degeneracy (lower=better)': [
            'repetition_rate_3gram', 'repetition_rate_5gram',
        ],
        'Distribution shape': ['zipf_coefficient'],
        'BLEU vs reference': ['bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'],
    }

    for group_name, keys in groups.items():
        present = [k for k in keys if k in metrics]
        if not present:
            continue
        lines.append(f"\n  [{group_name}]")
        for k in present:
            v = metrics[k]
            if isinstance(v, float):
                if v > 100:
                    lines.append(f"    {k:<28} {v:>12.2f}")
                elif v > 1:
                    lines.append(f"    {k:<28} {v:>12.4f}")
                else:
                    lines.append(f"    {k:<28} {v:>12.6f}")
            else:
                lines.append(f"    {k:<28} {v:>12}")

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)
