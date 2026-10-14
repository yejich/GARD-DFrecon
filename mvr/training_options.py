"""Shared CLI options for DA3 training and its CPU configuration preflight."""
import argparse
from omegaconf import OmegaConf


def boolean(value):
    if value not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Expected true or false')
    return value == 'true'


def epoch_list(value):
    try:
        epochs = [int(x) for x in value.split(',') if x.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected comma-separated epoch numbers') from error
    if any(e < 1 for e in epochs):
        raise argparse.ArgumentTypeError('Epoch numbers must be positive')
    return sorted(set(epochs))


def add_training_options(parser):
    parser.add_argument('--distractor-aware-loss', type=boolean)
    parser.add_argument('--epochs', type=int, help='Training epochs; LR holds for first half then decays')
    parser.add_argument('--global-batch-size', type=int)
    parser.add_argument('--grad-accum-steps', type=int)
    parser.add_argument('--original-groups-per-epoch', type=int)
    parser.add_argument('--log-every-optimizer-steps', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--save-epochs', type=epoch_list)
    parser.add_argument('--eval-epochs', type=epoch_list)
    parser.add_argument('--ema-only', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-enabled', type=boolean)
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-run-name', '--exp-name', dest='wandb_run_name')
    parser.add_argument('--wandb-entity')
    parser.add_argument('--manifest')
    parser.add_argument('--negative-manifest')
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--result-root')


def apply_training_options(cfg, args):
    mapping = {
        'distractor_aware_loss': 'mvrm.loss.attn_align.distractor_aware.use',
        'epochs': 'training.epochs', 'global_batch_size': 'training.global_batch_size',
        'grad_accum_steps': 'training.grad_accum_steps', 'num_workers': 'training.num_workers',
        'original_groups_per_epoch': 'data.train.pairs.original_groups_per_epoch',
        'log_every_optimizer_steps': 'training.log_interval',
        'save_epochs': 'training.save_weight_epochs', 'eval_epochs': 'fixed_eval.epochs',
        'manifest': 'data.train.pairs.manifest', 'negative_manifest': 'data.train.pairs.negative_manifest',
        'init_checkpoint': 'stage_2.ckpt', 'result_root': 'log.result_root_dir',
        'wandb_project': 'log.tracker.wandb.project', 'wandb_run_name': 'log.tracker.wandb.run_name',
        'wandb_entity': 'log.tracker.wandb.entity',
    }
    for arg, path in mapping.items():
        value = getattr(args, arg, None)
        if value is not None:
            OmegaConf.update(cfg, path, value if arg != 'wandb_entity' else value or None)
    if args.epochs is not None:
        cfg.training.scheduler.decay_end_epoch = args.epochs
        cfg.training.scheduler.warmup_epochs = args.epochs // 2
    if args.ema_only:
        cfg.training.save_optimizer_state = False
    enabled = args.wandb_enabled if args.wandb_enabled is not None else (True if args.wandb else None)
    if enabled is not None:
        cfg.log.tracker.name = 'wandb' if enabled else None
        if enabled:
            cfg.log.tracker.wandb.unique_run = True
            if not cfg.log.tracker.wandb.project:
                cfg.log.tracker.wandb.project = 'cross-view-feature-completion'
    for key in ('epochs', 'global_batch_size', 'grad_accum_steps', 'log_interval'):
        if int(cfg.training[key]) < 1:
            raise ValueError(f'training.{key} must be positive')
    if cfg.training.num_workers < 0 or cfg.data.train.pairs.get('original_groups_per_epoch', 0) < 0:
        raise ValueError('Workers and original groups must be nonnegative')
    for epochs in (cfg.training.get('save_weight_epochs', []), cfg.get('fixed_eval', {}).get('epochs', [])):
        if any(e < 1 or e > cfg.training.epochs for e in epochs):
            raise ValueError('Save/eval epochs must be within the training epoch range')
    return cfg
