import json
import os
import shutil
import time
import warnings

import pandas as pd
import torch
from core.data import get_data_info, load_data, SEMISUP_DATASETS
from core.utils import format_time, Logger, parser_train, seed
from gowal21uncovering.utils import WATrainerMamba




def load_mamba_checkpoint(trainer, path, logger):
    """Load a raw or WA-format MambaVision checkpoint into live and WA models."""
    raw = torch.load(path, map_location='cpu')

    if not isinstance(raw, dict):
        raise ValueError(f'Expected dict checkpoint, got {type(raw)}')

    if 'model_state_dict' in raw:
        sd = raw['model_state_dict']
        logger.log('Checkpoint format: WATrainer (model_state_dict)')
    elif all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
        logger.log('Checkpoint format: plain state dict')
    else:
        candidates = {k: v for k, v in raw.items() if isinstance(v, dict)}
        if not candidates:
            raise ValueError(f'Unrecognised checkpoint format. Keys: {list(raw.keys())}')
        key = next(iter(candidates))
        sd = candidates[key]
        logger.log(f'Checkpoint format: dict (using key "{key}")')

    # Normalize common wrapper prefixes.
    keys = list(sd.keys())
    if not keys:
        raise ValueError(f'Checkpoint contains an empty state dict: {path}')
    first_key = keys[0]
    for prefix in ('module.0.', 'module.', ''):
        if prefix == '' or first_key.startswith(prefix):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
            break

    if not any(k.startswith('mamba.') for k in sd):
        sd = {'mamba.' + k: v for k, v in sd.items()}
        logger.log('Added "mamba." prefix (raw model -> MambaVisionWrapper)')

    wrapper_model = trainer.model.module[0]
    wrapper_wa = trainer.wa_model.module[0]

    missing, unexpected = wrapper_model.load_state_dict(sd, strict=False)
    missing_wa, unexpected_wa = wrapper_wa.load_state_dict(sd, strict=False)

    if missing or unexpected or missing_wa or unexpected_wa:
        logger.log(
            f'Checkpoint load mismatch: live missing={len(missing)}, live unexpected={len(unexpected)}, '
            f'WA missing={len(missing_wa)}, WA unexpected={len(unexpected_wa)}'
        )
        logger.log(f'  live missing: {missing[:5]}')
        logger.log(f'  live unexpected: {unexpected[:5]}')
        if set(missing) != set(missing_wa) or set(unexpected) != set(unexpected_wa):
            raise RuntimeError('Live/WA checkpoint loading produced different key mismatches.')
    else:
        logger.log('State dict loaded cleanly into live and WA models.')

    logger.log(f'Loaded Mamba checkpoint: {path}')


def restore_best_from_history(trainer, resume_path, logger):
    """Restore best-score metadata for old checkpoints that predate this fix."""
    state_path = os.path.join(resume_path, 'state-last.pt')
    checkpoint = torch.load(state_path, map_location='cpu')
    if 'best_eval_adv_acc' in checkpoint:
        trainer.best_eval_adv_acc = float(checkpoint['best_eval_adv_acc'])
        trainer.best_clean_acc = float(checkpoint.get('best_clean_acc', 0.0))
        return

    stats_path = os.path.join(resume_path, 'stats_adv.csv')
    if os.path.exists(stats_path):
        stats = pd.read_csv(stats_path)
        if 'eval_adversarial_acc' in stats.columns and not stats.empty:
            valid = stats['eval_adversarial_acc'].dropna()
            if len(valid):
                best_idx = valid.idxmax()
                trainer.best_eval_adv_acc = float(stats.loc[best_idx, 'eval_adversarial_acc'])
                if 'test_clean_acc' in stats.columns:
                    trainer.best_clean_acc = float(stats.loc[best_idx, 'test_clean_acc'])
                logger.log(
                    f'Restored historical best from stats_adv.csv: '
                    f'eval_adv={trainer.best_eval_adv_acc:.6f}, clean={trainer.best_clean_acc:.6f}'
                )


def main():
    warnings.filterwarnings('once')

    parse = parser_train()
    parse.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')
    parse.add_argument('--num-workers', type=int, default=4, help='Number of dataloader workers.')
    parse.add_argument('--input-size', type=int, default=224, help='Spatial size used for MambaVision inputs.')
    parse.add_argument(
        '--mamba-checkpoint',
        type=str,
        default=None,
        help='Optional MambaVision checkpoint used to initialize live and WA models before adversarial training.',
    )
    parse.set_defaults(
        lr=5e-4,
        weight_decay=0.05,
        adv_eval_freq=5,
    )
    args = parse.parse_args()
    assert args.data in SEMISUP_DATASETS, f'Only data in {SEMISUP_DATASETS} is supported!'

    data_dir = os.path.join(args.data_dir, args.data)
    log_dir = os.path.join(args.log_dir, args.desc)
    weights = os.path.join(log_dir, 'weights-best.pt')

    if not args.resume_path and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(os.path.join(log_dir, 'log-train.log'))
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    info = get_data_info(data_dir)
    batch_size = args.batch_size
    batch_size_validation = args.batch_size_validation
    num_adv_epochs = args.num_adv_epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')
    if args.debug:
        num_adv_epochs = 1

    torch.backends.cudnn.benchmark = True

    seed(args.seed)
    aux_file = args.aux_data_filename if args.unsup_fraction > 0.0 else None
    (
        train_dataset,
        test_dataset,
        eval_dataset,
        train_dataloader,
        test_dataloader,
        eval_dataloader,
    ) = load_data(
        data_dir,
        batch_size,
        batch_size_validation,
        use_augmentation=args.augment,
        use_consistency=args.consistency,
        shuffle_train=True,
        aux_data_filename=aux_file,
        unsup_fraction=args.unsup_fraction,
        validation=True,
        num_workers=args.num_workers,
    )
    del train_dataset, test_dataset, eval_dataset

    # Design 2: keep dataloader tensors at native CIFAR resolution (32x32).
    # WATrainerMamba generates adversarial examples in this original input space.
    # MambaVisionWrapper performs the differentiable 32x32 -> input_size resize
    # inside the model forward pass.
    logger.log(
        f'Design 2: attack remains at native data resolution; '
        f'Mamba backbone input size: {args.input_size}x{args.input_size}'
    )

    seed(args.seed)
    trainer = WATrainerMamba(info, args)
    last_lr = args.lr

    if args.mamba_checkpoint and not args.resume_path:
        load_mamba_checkpoint(trainer, args.mamba_checkpoint, logger)

    if num_adv_epochs <= 0:
        logger.log('No adversarial epochs requested.')
        logger.log('Script Completed.')
        return

    metrics_path = os.path.join(log_dir, 'stats_adv.csv')
    metrics = pd.read_csv(metrics_path) if args.resume_path and os.path.exists(metrics_path) else pd.DataFrame()

    logger.log('\n\n')
    logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(trainer.eval(test_dataloader) * 100))

    trainer.init_optimizer(num_adv_epochs)
    test_adv_acc = 0.0

    if args.resume_path:
        resume_state = os.path.join(args.resume_path, 'state-last.pt')
        loaded_epoch = trainer.load_model_resume(resume_state)
        restore_best_from_history(trainer, args.resume_path, logger)
        start_epoch = loaded_epoch + 1
        logger.log(
            f'Resuming from completed epoch {loaded_epoch}; starting epoch {start_epoch}; '
            f'best eval adv={trainer.best_eval_adv_acc:.6f}'
        )
    else:
        trainer.best_clean_acc = 0.0
        trainer.best_eval_adv_acc = 0.0
        start_epoch = 1
        logger.log('Starting from epoch 1')

    if start_epoch > num_adv_epochs:
        logger.log(
            f'Resume checkpoint is already at epoch {start_epoch - 1}, '
            f'which is >= requested num_adv_epochs={num_adv_epochs}. Nothing to train.'
        )
        logger.log('Script Completed.')
        return

    for epoch in range(start_epoch, num_adv_epochs + 1):
        start = time.time()
        logger.log(f'======= Epoch {epoch} =======')

        if trainer.scheduler is not None:
            last_lr = trainer.scheduler.get_last_lr()[0]

        res = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
        test_acc = trainer.eval(test_dataloader)

        logger.log('Loss: {:.4f}.\tLR: {:.2e}'.format(res['loss'], last_lr))
        if 'clean_acc' in res:
            logger.log(
                'Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(
                    res['clean_acc'] * 100, test_acc * 100
                )
            )
        else:
            logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc * 100))

        epoch_metrics = {'train_' + k: v for k, v in res.items()}
        epoch_metrics.update(
            {
                'epoch': epoch,
                'lr': last_lr,
                'test_clean_acc': test_acc,
                'test_adversarial_acc': '',
            }
        )

        if epoch % args.adv_eval_freq == 0 or epoch == num_adv_epochs:
            test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
            logger.log(
                'Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(
                    res['adversarial_acc'] * 100, test_adv_acc * 100
                )
            )
            epoch_metrics['test_adversarial_acc'] = test_adv_acc
        else:
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc'] * 100))

        eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
        logger.log('Adversarial Accuracy-\tEval: {:.2f}%.'.format(eval_adv_acc * 100))
        epoch_metrics['eval_adversarial_acc'] = eval_adv_acc

        if eval_adv_acc >= trainer.best_eval_adv_acc:
            trainer.best_clean_acc = test_acc
            trainer.best_eval_adv_acc = eval_adv_acc
            trainer.save_model(weights)
            logger.log(
                f'New best checkpoint: eval_adv={trainer.best_eval_adv_acc * 100:.2f}%, '
                f'test_clean={trainer.best_clean_acc * 100:.2f}%'
            )

        trainer.save_model_resume(os.path.join(log_dir, 'state-last.pt'), epoch)

        if epoch % 400 == 0 and os.path.exists(weights):
            shutil.copyfile(weights, os.path.join(log_dir, f'weights-best-epoch{epoch}.pt'))

        logger.log('Time taken: {}'.format(format_time(time.time() - start)))
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(metrics_path, index=False)

    train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
    logger.log('\nTraining completed.')
    logger.log(
        'Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(
            train_acc * 100, trainer.best_clean_acc * 100
        )
    )
    logger.log(
        'Adversarial Accuracy-\tTrain: {:.2f}%.\tEval: {:.2f}%.'.format(
            res['adversarial_acc'] * 100, trainer.best_eval_adv_acc * 100
        )
    )
    logger.log('Script Completed.')


if __name__ == '__main__':
    main()
