"""
LD3-D: Learning to Discretize Discrete Diffusion.

Main entry point for training optimal timestep schedules for discrete
diffusion models (MDLM, SEDD, D3PM, Discrete Flow Matching).

Usage:
    # Generate teacher data (placeholder for testing)
    python discrete_gen_data.py --placeholder --output_dir data/discrete_teacher/test

    # Train LD3-D schedule
    python main_discrete.py --config configs/discrete/mdlm_text8.yml

    # Evaluate with learned schedule
    python main_discrete.py --config configs/discrete/mdlm_text8.yml --eval_only --load_from logs/discrete/mdlm_text8/best_discrete.pt
"""

import argparse
import logging
import os
import yaml
import torch

from discrete_noise_schedulers import AbsorbingSchedule, UniformSchedule, DiscreteFlowSchedule
from discrete_samplers import TauLeapingSolver, AnalyticSolver
from discrete_trainer import (
    LD3DTrainer, DiscreteModelConfig, DiscreteTrainingConfig, DiscreteLD3Dataset
)
from discrete_gen_data import load_training_data


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)


def build_noise_schedule(config):
    """Build discrete noise schedule from config."""
    schedule_type = config.get('noise_schedule', 'log_linear')
    diffusion_type = config.get('diffusion_type', 'absorbing')
    vocab_size = config.get('vocab_size', 32000) + 1  # +1 for MASK token
    T = config.get('T', 1.0)
    eps = config.get('eps', 1e-4)

    if diffusion_type == 'absorbing':
        return AbsorbingSchedule(
            vocab_size=vocab_size,
            schedule=schedule_type,
            T=T, eps=eps
        )
    elif diffusion_type == 'uniform':
        return UniformSchedule(
            vocab_size=vocab_size,
            schedule=schedule_type,
            T=T, eps=eps
        )
    elif diffusion_type == 'flow':
        return DiscreteFlowSchedule(
            vocab_size=vocab_size,
            schedule='linear',
            T=T, eps=eps
        )
    else:
        raise ValueError(f"Unknown diffusion_type: {diffusion_type}")


def build_solver(config, noise_schedule):
    """Build discrete sampler from config."""
    solver_name = config.get('solver_name', 'tau_leaping')
    diffusion_type = config.get('diffusion_type', 'absorbing')

    if solver_name == 'tau_leaping':
        return TauLeapingSolver(noise_schedule, diffusion_type)
    elif solver_name == 'analytic':
        return AnalyticSolver(noise_schedule, diffusion_type)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def build_model(config, device):
    """Build or load the pre-trained discrete diffusion model.

    For testing, returns a placeholder model. In practice, this would
    load MDLM, SEDD, D3PM, etc. from their respective checkpoints.
    """
    ckp_path = config.get('ckp_path')
    model_type = config.get('model_type', 'mdlm')
    vocab_size = config.get('vocab_size', 32000) + 1
    seq_length = config.get('seq_length', 256)

    if ckp_path and os.path.exists(ckp_path):
        logging.info(f"Loading pre-trained model from {ckp_path}")
        # Model-specific loading would go here
        # e.g., for MDLM: load MDLM checkpoint
        # For now, return placeholder
        pass

    logging.info("Using placeholder model for testing/development")
    return PlaceholderDiscreteModel(vocab_size, seq_length).to(device)


class PlaceholderDiscreteModel(torch.nn.Module):
    """Placeholder model for testing the LD3-D pipeline.

    Returns random logits. Replace with actual pre-trained model.
    """
    def __init__(self, vocab_size, seq_length, hidden_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.time_embed = torch.nn.Linear(1, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, t, condition=None):
        """
        Args:
            x: [B, L] token indices or [B, L, K] probabilities
            t: [B] or scalar
        Returns:
            logits: [B, L, K]
        """
        if x.dim() == 3:
            # Soft input: weighted embedding
            h = torch.matmul(x, self.embed.weight)  # [B, L, D]
        else:
            h = self.embed(x)  # [B, L, D]

        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=h.device).float()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.expand(h.shape[0])

        t_emb = self.time_embed(t.unsqueeze(-1).float())  # [B, D]
        h = h + t_emb.unsqueeze(1)  # [B, L, D]
        logits = self.proj(h)  # [B, L, K]
        return logits

    def soft(self, p, t, condition=None):
        """Soft-input forward for probability flow."""
        return self.forward(p, t, condition)


def main():
    parser = argparse.ArgumentParser(description='LD3-D: Learn Discrete Diffusion Schedules')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate, do not train')
    parser.add_argument('--load_from', type=str, default=None,
                        help='Path to checkpoint to load')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--placeholder_data', action='store_true',
                        help='Generate and use placeholder data for testing')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Build components
    noise_schedule = build_noise_schedule(config)
    solver = build_solver(config, noise_schedule)
    model = build_model(config, device)

    # Load or generate data
    data_dir = config.get('data_dir', 'data/discrete_teacher/test')

    if args.placeholder_data or not os.path.exists(os.path.join(data_dir, 'train.pt')):
        logging.info("Generating placeholder data...")
        from discrete_gen_data import _generate_placeholder_data
        os.makedirs(data_dir, exist_ok=True)
        _generate_placeholder_data(
            data_dir,
            config.get('num_train', 50),
            config.get('num_valid', 25),
            config.get('seq_length', 256),
            config.get('vocab_size', 32000) + 1,
            config.get('diffusion_type', 'absorbing'),
            device
        )

    train_dataset, valid_dataset = load_training_data(data_dir)

    # Build configs
    model_config = DiscreteModelConfig(
        net=model,
        noise_schedule=noise_schedule,
        solver=solver,
        diffusion_type=config.get('diffusion_type', 'absorbing'),
        steps=config.get('steps', 32),
        vocab_size=config.get('vocab_size', 32000) + 1,
        seq_length=config.get('seq_length', 256),
        time_mode=config.get('time_mode', 'time'),
        snapshot_path=config.get('snapshot_path', 'logs/discrete'),
        device=device,
    )

    training_config = DiscreteTrainingConfig(
        train_data=train_dataset,
        valid_data=valid_dataset,
        train_batch_size=config.get('train_batch_size', 8),
        valid_batch_size=config.get('valid_batch_size', 8),
        lr_time_1=config.get('lr_time_1', 0.005),
        lr_time_2=config.get('lr_time_2', 0.1),
        min_lr_time_1=config.get('min_lr_time_1', 5e-5),
        min_lr_time_2=config.get('min_lr_time_2', 1e-6),
        win_rate=config.get('win_rate', 0.5),
        patient=config.get('patient', 5),
        lr2_patient=config.get('lr2_patient', 5),
        lr_time_decay=config.get('lr_time_decay', 0.8),
        momentum_time_1=config.get('momentum_time_1', 0.9),
        loss_type=config.get('loss_type', 'cross_entropy'),
        theory_proxy_weight=config.get('theory_proxy_weight', 0.0),
        temperature=config.get('temperature', 1.0),
        training_rounds_v1=config.get('training_rounds_v1', 3),
        training_rounds_v2=config.get('training_rounds_v2', 5),
    )

    # Create trainer
    trainer = LD3DTrainer(model_config, training_config)

    if args.load_from:
        checkpoint = torch.load(args.load_from, map_location=device)
        trainer.params1.data = checkpoint['params1'].to(device)
        trainer.params2.data = checkpoint['params2'].to(device)
        logging.info(f"Loaded schedule from {args.load_from}")

    if args.eval_only:
        val_loss = trainer._run_validation()
        logging.info(f"Validation loss: {val_loss:.6f}")
        ts1, ts2 = trainer._get_schedule()
        logging.info(f"Schedule: {ts1.detach().cpu().tolist()}")
    else:
        trainer.train()

    logging.info("Done!")


if __name__ == '__main__':
    main()
