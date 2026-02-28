"""
Teacher data generation for LD3-D.

Generates high-quality samples from pre-trained discrete diffusion models
using many sampling steps (teacher), then stores them as training targets
for schedule optimization.

Usage:
    python discrete_gen_data.py --config configs/discrete/mdlm_text8.yml

The output is a directory of .pt files, each containing:
  - 'noise_tokens': Initial noise (all-mask or random) [L]
  - 'teacher_tokens': Clean tokens from teacher sampler [L]
  - 'condition': Optional conditioning embedding
"""

import argparse
import os
import logging
import torch
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def generate_noise_tokens(batch_size, seq_length, vocab_size, diffusion_type, device):
    """Generate initial noisy tokens for the forward process.

    Args:
        batch_size: Number of samples
        seq_length: Sequence length L
        vocab_size: Vocabulary size K
        diffusion_type: 'absorbing' or 'uniform'
        device: torch device

    Returns:
        noise_tokens: [B, L] initial tokens
    """
    if diffusion_type == 'absorbing':
        # All positions are MASK (last token index)
        mask_idx = vocab_size - 1
        return torch.full((batch_size, seq_length), mask_idx,
                          dtype=torch.long, device=device)
    elif diffusion_type == 'uniform':
        # Uniform random tokens
        return torch.randint(0, vocab_size, (batch_size, seq_length),
                             device=device)
    else:
        return torch.randint(0, vocab_size, (batch_size, seq_length),
                             device=device)


def generate_teacher_samples(model, noise_schedule, solver, noise_tokens,
                             teacher_steps, diffusion_type, device,
                             condition=None, skip_type='uniform'):
    """Generate high-quality samples using many-step teacher sampler.

    Args:
        model: Pre-trained score model
        noise_schedule: DiscreteNoiseSchedule
        solver: Discrete sampler instance
        noise_tokens: Initial noise [B, L]
        teacher_steps: Number of teacher steps (e.g., 1024)
        diffusion_type: 'absorbing' or 'uniform'
        device: torch device
        condition: Optional conditioning
        skip_type: Baseline schedule type

    Returns:
        teacher_tokens: Generated tokens [B, L]
    """
    # Create teacher schedule with many steps
    timesteps = solver.get_time_steps(
        skip_type=skip_type,
        t_T=noise_schedule.T,
        t_0=noise_schedule.eps,
        N=teacher_steps,
        device=device
    )

    with torch.no_grad():
        teacher_tokens = solver.sample_simple(
            model_fn=model,
            x=noise_tokens,
            timesteps=timesteps,
            timesteps2=timesteps,
            order=1,
            NFEs=teacher_steps,
            condition=condition,
        )

    return teacher_tokens


def generate_training_data(config):
    """Main function to generate LD3-D training data.

    Args:
        config: Configuration dict with model, data, and generation parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = config.get('output_dir', 'data/discrete_teacher')
    os.makedirs(output_dir, exist_ok=True)

    num_train = config.get('num_train', 50)
    num_valid = config.get('num_valid', 25)
    batch_size = config.get('batch_size', 8)
    seq_length = config.get('seq_length', 256)
    vocab_size = config.get('vocab_size', 32000)
    teacher_steps = config.get('teacher_steps', 1024)
    diffusion_type = config.get('diffusion_type', 'absorbing')
    skip_type = config.get('skip_type', 'uniform')

    # Load model and noise schedule (model-specific)
    model = config.get('model')
    noise_schedule = config.get('noise_schedule')
    solver = config.get('solver')

    if model is None:
        logging.warning("No model provided. Creating placeholder data for testing.")
        _generate_placeholder_data(
            output_dir, num_train, num_valid, seq_length, vocab_size,
            diffusion_type, device
        )
        return

    model.eval()
    total_samples = num_train + num_valid

    all_noise = []
    all_teacher = []
    all_conditions = []

    num_batches = (total_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating teacher data"):
        current_batch_size = min(batch_size, total_samples - batch_idx * batch_size)

        noise_tokens = generate_noise_tokens(
            current_batch_size, seq_length, vocab_size, diffusion_type, device
        )

        teacher_tokens = generate_teacher_samples(
            model, noise_schedule, solver, noise_tokens,
            teacher_steps, diffusion_type, device,
            skip_type=skip_type
        )

        all_noise.append(noise_tokens.cpu())
        all_teacher.append(teacher_tokens.cpu())

    # Concatenate and split
    all_noise = torch.cat(all_noise, dim=0)[:total_samples]
    all_teacher = torch.cat(all_teacher, dim=0)[:total_samples]

    # Save train and validation splits
    train_data = {
        'noise_tokens': all_noise[:num_train],
        'teacher_tokens': all_teacher[:num_train],
    }
    valid_data = {
        'noise_tokens': all_noise[num_train:num_train + num_valid],
        'teacher_tokens': all_teacher[num_train:num_train + num_valid],
    }

    torch.save(train_data, os.path.join(output_dir, 'train.pt'))
    torch.save(valid_data, os.path.join(output_dir, 'valid.pt'))

    logging.info(f"Generated {num_train} train + {num_valid} valid samples")
    logging.info(f"Saved to {output_dir}")


def _generate_placeholder_data(output_dir, num_train, num_valid, seq_length,
                                vocab_size, diffusion_type, device):
    """Generate placeholder data for testing the pipeline."""
    mask_idx = vocab_size - 1

    noise_train = generate_noise_tokens(
        num_train, seq_length, vocab_size, diffusion_type, device
    )
    noise_valid = generate_noise_tokens(
        num_valid, seq_length, vocab_size, diffusion_type, device
    )

    # Placeholder "teacher" tokens (random, for testing only)
    teacher_train = torch.randint(0, vocab_size - 1, (num_train, seq_length))
    teacher_valid = torch.randint(0, vocab_size - 1, (num_valid, seq_length))

    train_data = {
        'noise_tokens': noise_train.cpu(),
        'teacher_tokens': teacher_train,
    }
    valid_data = {
        'noise_tokens': noise_valid.cpu(),
        'teacher_tokens': teacher_valid,
    }

    torch.save(train_data, os.path.join(output_dir, 'train.pt'))
    torch.save(valid_data, os.path.join(output_dir, 'valid.pt'))
    logging.info(f"Generated placeholder data in {output_dir}")


def load_training_data(data_dir, device=None):
    """Load previously generated training data.

    Args:
        data_dir: Directory containing train.pt and valid.pt
        device: Optional device to load to

    Returns:
        train_dataset, valid_dataset: DiscreteLD3Dataset instances
    """
    from discrete_trainer import DiscreteLD3Dataset

    train_data = torch.load(os.path.join(data_dir, 'train.pt'), map_location='cpu')
    valid_data = torch.load(os.path.join(data_dir, 'valid.pt'), map_location='cpu')

    # Convert to per-sample lists
    train_noise = [train_data['noise_tokens'][i] for i in range(len(train_data['noise_tokens']))]
    train_teacher = [train_data['teacher_tokens'][i] for i in range(len(train_data['teacher_tokens']))]
    valid_noise = [valid_data['noise_tokens'][i] for i in range(len(valid_data['noise_tokens']))]
    valid_teacher = [valid_data['teacher_tokens'][i] for i in range(len(valid_data['teacher_tokens']))]

    train_conds = None
    valid_conds = None
    if 'conditions' in train_data and train_data['conditions'] is not None:
        train_conds = [train_data['conditions'][i] for i in range(len(train_data['conditions']))]
        valid_conds = [valid_data['conditions'][i] for i in range(len(valid_data['conditions']))]

    train_dataset = DiscreteLD3Dataset(train_noise, train_teacher, train_conds)
    valid_dataset = DiscreteLD3Dataset(valid_noise, valid_teacher, valid_conds)

    return train_dataset, valid_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LD3-D training data')
    parser.add_argument('--output_dir', type=str, default='data/discrete_teacher')
    parser.add_argument('--num_train', type=int, default=50)
    parser.add_argument('--num_valid', type=int, default=25)
    parser.add_argument('--seq_length', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--diffusion_type', type=str, default='absorbing')
    parser.add_argument('--teacher_steps', type=int, default=1024)
    parser.add_argument('--placeholder', action='store_true',
                        help='Generate placeholder data for testing')

    args = parser.parse_args()

    if args.placeholder:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _generate_placeholder_data(
            args.output_dir, args.num_train, args.num_valid,
            args.seq_length, args.vocab_size, args.diffusion_type, device
        )
    else:
        config = vars(args)
        config['model'] = None  # Placeholder
        generate_training_data(config)
