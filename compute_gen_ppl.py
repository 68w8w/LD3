"""
Generative Perplexity (Gen PPL) evaluation for LD3-D.

Standard evaluation protocol for discrete diffusion text generation:
  1. Generate N text samples using the discrete diffusion model + learned schedule
  2. Decode tokens to text strings
  3. Re-tokenize with GPT-2's tokenizer
  4. Compute perplexity under GPT-2 Large (or other AR model)

This follows the exact protocol used by MDLM (Sahoo et al., NeurIPS 2024)
and SEDD (Lou et al., ICML 2024).

Usage:
    python compute_gen_ppl.py \
        --config configs/discrete/mdlm_text8.yml \
        --load_from logs/discrete/mdlm_text8/best_discrete.pt \
        --eval_model gpt2-large \
        --num_samples 200 \
        --seq_length 1024

Reference:
  - MDLM: github.com/kuleshov-group/mdlm (compute_generative_perplexity)
  - SEDD: Lou et al. (ICML 2024 Best Paper)
  - LSD: Liu et al. (2025)
"""

import argparse
import logging
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# ============================================================================
# Core Gen PPL Computation (follows MDLM protocol)
# ============================================================================

def compute_generative_perplexity(
    generated_token_ids,
    eval_model_name='gpt2-large',
    tokenizer_name=None,
    source_tokenizer=None,
    eval_batch_size=8,
    eval_context_size=1024,
    device=None,
):
    """Compute Generative Perplexity of generated samples under an AR model.

    This is the standard evaluation metric for discrete diffusion text generation,
    as used by MDLM, SEDD, BD3-LM, and LSD.

    Protocol:
      1. Convert generated token IDs -> text (using source tokenizer)
      2. Re-tokenize text with the eval model's tokenizer (GPT-2 Large)
      3. Compute per-token cross-entropy under the eval model
      4. PPL = exp(mean CE over all tokens)

    Args:
        generated_token_ids: List of 1D tensors or a 2D tensor [N, L],
            token indices from the discrete diffusion model's vocabulary.
        eval_model_name: HuggingFace model name for the evaluator.
            Default: 'gpt2-large'. Also supports 'gpt2-xl', 'meta-llama/Llama-2-7b-hf'.
        tokenizer_name: Tokenizer for eval model. If None, uses eval_model_name.
        source_tokenizer: Tokenizer used by the diffusion model (to decode token IDs to text).
            If None, assumes token IDs can be directly decoded via chr() (e.g., text8).
        eval_batch_size: Batch size for GPT-2 evaluation.
        eval_context_size: Context window for eval model (1024 for GPT-2, 4096 for Llama2).
        device: Torch device.

    Returns:
        gen_ppl: Generative perplexity (float). Lower = better quality.
        results: Dict with detailed metrics.
    """
    import transformers

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if tokenizer_name is None:
        tokenizer_name = eval_model_name

    # ------------------------------------------------------------------
    # Step 1: Load evaluation model (GPT-2 Large by default)
    # ------------------------------------------------------------------
    logging.info(f"Loading eval model: {eval_model_name}")
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
        eval_model_name
    ).eval().to(device)

    eval_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    if eval_tokenizer.pad_token is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    # Adjust context size for model type
    if 'llama' in eval_model_name.lower():
        eval_context_size = min(eval_context_size, 4096)
    else:
        eval_context_size = min(eval_context_size, 1024)

    # ------------------------------------------------------------------
    # Step 2: Decode generated tokens to text strings
    # ------------------------------------------------------------------
    logging.info("Decoding generated samples to text...")
    texts = decode_samples_to_text(generated_token_ids, source_tokenizer)
    logging.info(f"  {len(texts)} samples, avg length: {np.mean([len(t) for t in texts]):.0f} chars")

    # ------------------------------------------------------------------
    # Step 3: Re-tokenize with eval model's tokenizer
    # ------------------------------------------------------------------
    logging.info("Re-tokenizing with eval model tokenizer...")
    eval_encodings = eval_tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=eval_context_size,
    )
    input_ids = eval_encodings['input_ids'].to(device)       # [N, L_eval]
    attention_mask = eval_encodings['attention_mask'].to(device)  # [N, L_eval]

    N, L_eval = input_ids.shape
    logging.info(f"  Re-tokenized: {N} samples x {L_eval} tokens")

    # ------------------------------------------------------------------
    # Step 4: Compute per-token cross-entropy under eval model
    # ------------------------------------------------------------------
    logging.info(f"Computing Gen PPL with {eval_model_name}...")
    total_ce = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in tqdm(range(0, N, eval_batch_size), desc="Gen PPL"):
            end = min(start + eval_batch_size, N)
            batch_ids = input_ids[start:end]            # [B, L]
            batch_mask = attention_mask[start:end]       # [B, L]

            # Handle sequences longer than context size by chunking
            for chunk_start in range(0, batch_ids.shape[1], eval_context_size):
                chunk_end = min(chunk_start + eval_context_size, batch_ids.shape[1])
                chunk_ids = batch_ids[:, chunk_start:chunk_end]
                chunk_mask = batch_mask[:, chunk_start:chunk_end]

                if chunk_ids.shape[1] < 2:
                    continue

                outputs = eval_model(chunk_ids, attention_mask=chunk_mask)
                logits = outputs.logits  # [B, L, V_eval]

                # Shift: predict token i+1 from token i
                shift_logits = logits[:, :-1, :].contiguous()      # [B, L-1, V]
                shift_labels = chunk_ids[:, 1:].contiguous()        # [B, L-1]
                shift_mask = chunk_mask[:, 1:].contiguous().float() # [B, L-1]

                # Per-token cross-entropy
                ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none'
                ).view(shift_labels.shape)  # [B, L-1]

                # Mask out padding tokens
                ce = ce * shift_mask
                total_ce += ce.sum().item()
                total_tokens += shift_mask.sum().item()

    # ------------------------------------------------------------------
    # Step 5: Compute final PPL
    # ------------------------------------------------------------------
    avg_ce = total_ce / max(total_tokens, 1)
    gen_ppl = math.exp(min(avg_ce, 100))  # Clamp to avoid overflow

    results = {
        'gen_ppl': gen_ppl,
        'avg_ce': avg_ce,
        'total_tokens': int(total_tokens),
        'num_samples': N,
        'eval_model': eval_model_name,
    }

    logging.info(f"=" * 50)
    logging.info(f"Generative PPL ({eval_model_name}): {gen_ppl:.2f}")
    logging.info(f"  Avg CE: {avg_ce:.4f}")
    logging.info(f"  Total tokens scored: {total_tokens}")
    logging.info(f"=" * 50)

    # Cleanup
    del eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return gen_ppl, results


def decode_samples_to_text(token_ids, source_tokenizer=None):
    """Decode discrete diffusion token IDs to text strings.

    Args:
        token_ids: [N, L] tensor or list of 1D tensors.
        source_tokenizer: The diffusion model's tokenizer.
            - If a HuggingFace tokenizer: uses .decode()
            - If 'text8': maps 0-25 -> a-z, 26 -> space
            - If None: falls back to text8 mapping

    Returns:
        texts: List of N strings.
    """
    if isinstance(token_ids, torch.Tensor) and token_ids.dim() == 2:
        samples = [token_ids[i] for i in range(token_ids.shape[0])]
    else:
        samples = token_ids

    texts = []
    for sample in samples:
        if isinstance(sample, torch.Tensor):
            sample = sample.cpu().tolist()

        if source_tokenizer is None or source_tokenizer == 'text8':
            # text8 encoding: 0=a, 1=b, ..., 25=z, 26=space
            text = _decode_text8(sample)
        elif isinstance(source_tokenizer, str) and source_tokenizer == 'bytes':
            # Byte-level encoding
            text = bytes(sample).decode('utf-8', errors='replace')
        elif hasattr(source_tokenizer, 'decode'):
            # HuggingFace tokenizer
            # Filter out special tokens (mask, pad, etc.)
            filtered = [t for t in sample if t < source_tokenizer.vocab_size]
            text = source_tokenizer.decode(filtered, skip_special_tokens=True)
        else:
            text = _decode_text8(sample)

        texts.append(text)

    return texts


def _decode_text8(token_ids):
    """Decode text8-style token IDs to string.

    Mapping: 0-25 -> 'a'-'z', 26 -> ' ' (space).
    Any ID >= 27 (e.g., MASK token) is skipped.
    """
    chars = []
    for t in token_ids:
        if 0 <= t <= 25:
            chars.append(chr(ord('a') + t))
        elif t == 26:
            chars.append(' ')
        # Skip MASK tokens and other special tokens
    return ''.join(chars)


# ============================================================================
# Sample generation with learned schedule (actual discrete sampling)
# ============================================================================

def generate_samples_with_schedule(
    model, noise_schedule, solver, schedule_path,
    num_samples=200, seq_length=1024, batch_size=8,
    diffusion_type='absorbing', skip_type='uniform',
    device=None, condition=None,
):
    """Generate actual discrete token samples using a learned LD3-D schedule.

    Unlike probability-flow tracking (used during training), this performs
    real discrete sampling (tau-leaping or analytic) for evaluation.

    Args:
        model: Pre-trained discrete diffusion score model.
        noise_schedule: DiscreteNoiseSchedule.
        solver: DiscreteSolver (TauLeapingSolver or AnalyticSolver).
        schedule_path: Path to LD3-D checkpoint (.pt) containing learned timesteps.
            If None, uses baseline uniform schedule.
        num_samples: Number of samples to generate.
        seq_length: Sequence length per sample.
        batch_size: Generation batch size.
        diffusion_type: 'absorbing' or 'uniform'.
        skip_type: Baseline schedule type (used if schedule_path is None).
        device: Torch device.
        condition: Optional conditioning tensor.

    Returns:
        all_samples: Tensor [num_samples, seq_length] of generated token IDs.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    K = noise_schedule.vocab_size

    # Load learned schedule or use baseline
    if schedule_path and os.path.exists(schedule_path):
        checkpoint = torch.load(schedule_path, map_location=device)
        all_steps = checkpoint['best_t_steps']
        N = all_steps.shape[0] // 2
        timesteps1 = all_steps[:N].to(device)
        timesteps2 = all_steps[N:].to(device)
        logging.info(f"Using learned schedule ({N-1} steps) from {schedule_path}")
    else:
        steps = 32  # Default
        timesteps1 = solver.get_time_steps(
            skip_type=skip_type,
            t_T=noise_schedule.T,
            t_0=noise_schedule.eps,
            N=steps,
            device=device
        )
        timesteps2 = timesteps1
        logging.info(f"Using baseline {skip_type} schedule ({steps} steps)")

    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            cur_batch = min(batch_size, num_samples - batch_idx * batch_size)

            # Initialize noise
            if diffusion_type == 'absorbing':
                mask_idx = noise_schedule.mask_index
                x_init = torch.full(
                    (cur_batch, seq_length), mask_idx,
                    dtype=torch.long, device=device
                )
            else:
                x_init = torch.randint(
                    0, K, (cur_batch, seq_length), device=device
                )

            # Sample using the discrete solver
            x_gen = solver.sample_simple(
                model_fn=model,
                x=x_init,
                timesteps=timesteps1,
                timesteps2=timesteps2,
                order=1,
                condition=condition,
            )

            all_samples.append(x_gen.cpu())

    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    logging.info(f"Generated {all_samples.shape[0]} samples of length {seq_length}")

    return all_samples


# ============================================================================
# Full evaluation pipeline
# ============================================================================

def evaluate_gen_ppl(
    model, noise_schedule, solver,
    schedule_path=None,
    eval_model_name='gpt2-large',
    source_tokenizer=None,
    num_samples=200,
    seq_length=1024,
    batch_size=8,
    diffusion_type='absorbing',
    device=None,
    baselines=None,
):
    """Full Gen PPL evaluation: generate + score.

    Evaluates the learned schedule and optionally compares with baselines.

    Args:
        model: Discrete diffusion model.
        noise_schedule: DiscreteNoiseSchedule.
        solver: DiscreteSolver.
        schedule_path: Path to LD3-D checkpoint.
        eval_model_name: GPT-2 variant for scoring.
        source_tokenizer: Tokenizer for decoding diffusion tokens.
        num_samples: Number of samples to generate per schedule.
        seq_length: Sequence length.
        batch_size: Batch size for both generation and eval.
        diffusion_type: 'absorbing' or 'uniform'.
        device: Torch device.
        baselines: List of (name, schedule_path_or_None, skip_type) to compare.

    Returns:
        results: Dict mapping schedule_name -> gen_ppl.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    # Evaluate learned schedule
    if schedule_path:
        logging.info("=" * 60)
        logging.info("Evaluating LEARNED schedule (LD3-D)")
        logging.info("=" * 60)
        samples = generate_samples_with_schedule(
            model, noise_schedule, solver, schedule_path,
            num_samples=num_samples, seq_length=seq_length,
            batch_size=batch_size, diffusion_type=diffusion_type,
            device=device,
        )
        gen_ppl, detail = compute_generative_perplexity(
            samples, eval_model_name=eval_model_name,
            source_tokenizer=source_tokenizer,
            eval_batch_size=batch_size, device=device,
        )
        results['Learned (LD3-D)'] = gen_ppl

    # Evaluate baselines
    if baselines is None:
        baselines = [
            ('Uniform (time)', None, 'uniform'),
            ('Uniform (rate-integral)', None, 'rate_uniform'),
            ('Quadratic', None, 'quadratic'),
        ]

    for name, sched_path, skip_type in baselines:
        logging.info("=" * 60)
        logging.info(f"Evaluating baseline: {name}")
        logging.info("=" * 60)
        samples = generate_samples_with_schedule(
            model, noise_schedule, solver, sched_path,
            num_samples=num_samples, seq_length=seq_length,
            batch_size=batch_size, diffusion_type=diffusion_type,
            skip_type=skip_type, device=device,
        )
        gen_ppl, detail = compute_generative_perplexity(
            samples, eval_model_name=eval_model_name,
            source_tokenizer=source_tokenizer,
            eval_batch_size=batch_size, device=device,
        )
        results[name] = gen_ppl

    # Print comparison table
    logging.info("\n" + "=" * 60)
    logging.info(f"{'GENERATIVE PPL COMPARISON':^60}")
    logging.info(f"{'(eval model: ' + eval_model_name + ')':^60}")
    logging.info("=" * 60)
    logging.info(f"  {'Schedule':<30} {'Gen PPL':>10}")
    logging.info("-" * 60)
    for name, ppl in sorted(results.items(), key=lambda x: x[1]):
        marker = " <-- best" if ppl == min(results.values()) else ""
        logging.info(f"  {name:<30} {ppl:>10.2f}{marker}")
    logging.info("=" * 60)

    if 'Learned (LD3-D)' in results:
        learned_ppl = results['Learned (LD3-D)']
        baseline_ppls = {k: v for k, v in results.items() if k != 'Learned (LD3-D)'}
        if baseline_ppls:
            best_baseline = min(baseline_ppls.values())
            improvement = (best_baseline - learned_ppl) / best_baseline * 100
            logging.info(f"  LD3-D improvement over best baseline: {improvement:+.2f}%")

    return results


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Generative PPL for LD3-D')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--load_from', type=str, default=None,
                        help='Path to LD3-D checkpoint with learned schedule')
    parser.add_argument('--eval_model', type=str, default='gpt2-large',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='AR model for computing Gen PPL')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples to generate')
    parser.add_argument('--seq_length', type=int, default=1024,
                        help='Sequence length per sample')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--source_tokenizer', type=str, default='text8',
                        choices=['text8', 'bytes', 'gpt2'],
                        help='Tokenizer used by the diffusion model')
    parser.add_argument('--diffusion_type', type=str, default='absorbing')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # For standalone testing: generate random tokens and compute Gen PPL
    if args.config is None:
        logging.info("No config provided. Running standalone Gen PPL test with random tokens...")
        vocab_size = 28  # text8
        fake_samples = torch.randint(0, 27, (args.num_samples, args.seq_length))
        gen_ppl, results = compute_generative_perplexity(
            fake_samples,
            eval_model_name=args.eval_model,
            source_tokenizer=args.source_tokenizer,
            eval_batch_size=args.batch_size,
            device=device,
        )
    else:
        # Full evaluation with config
        import yaml
        from main_discrete import build_noise_schedule, build_solver, build_model

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        noise_schedule = build_noise_schedule(config)
        solver = build_solver(config, noise_schedule)
        model = build_model(config, device)

        results = evaluate_gen_ppl(
            model, noise_schedule, solver,
            schedule_path=args.load_from,
            eval_model_name=args.eval_model,
            source_tokenizer=args.source_tokenizer,
            num_samples=args.num_samples,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            diffusion_type=config.get('diffusion_type', 'absorbing'),
            device=device,
        )
