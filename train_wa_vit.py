import json
import os
import shutil
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch

from core.data import get_data_info, load_data, SEMISUP_DATASETS
from core.utils import format_time, Logger, parser_train, seed
from gowal21uncovering.utils import WATrainerViT2



def load_vit_checkpoint(trainer, path, logger):
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

    keys = list(sd.keys())
    if not keys:
        raise ValueError(f'Empty checkpoint state dict: {path}')
    first = keys[0]
    for prefix in ('module.0.', 'module.', ''):
        if prefix == '' or first.startswith(prefix):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
            break

    model = trainer.model.module[0]
    wa_model = trainer.wa_model.module[0]

    # Design-2 wrapper stores the actual VisionTransformer under ``vit``.
    # Accept both raw VisionTransformer checkpoints and wrapper checkpoints.
    model_has_vit_prefix = any(k.startswith('vit.') for k in model.state_dict())
    ckpt_has_vit_prefix = any(k.startswith('vit.') for k in sd)
    if model_has_vit_prefix and not ckpt_has_vit_prefix:
        sd = {'vit.' + k: v for k, v in sd.items()}
        logger.log('Added "vit." prefix (raw VisionTransformer -> ViT224Wrapper2)')

    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing_wa, unexpected_wa = wa_model.load_state_dict(sd, strict=False)

    # Wrapper/model buffers may legitimately be absent from older checkpoints.
    missing_real = [k for k in missing if not k.endswith('matrixMean') and not k.endswith('matrixStd')]
    missing_wa_real = [k for k in missing_wa if not k.endswith('matrixMean') and not k.endswith('matrixStd')]
    if missing_real or unexpected or missing_wa_real or unexpected_wa:
        raise RuntimeError(
            f'ViT checkpoint mismatch: live missing={missing_real}, live unexpected={unexpected}; '
            f'WA missing={missing_wa_real}, WA unexpected={unexpected_wa}'
        )
    logger.log(f'Loaded ViT checkpoint: {path}')


def main():
    parse = parser_train()
    parse.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')
    parse.add_argument('--vit-checkpoint', type=str, default=None,
                       help='Clean ViT-B/ViT-L CIFAR-10 checkpoint used to initialize adversarial training.')
    parse.add_argument('--input-size', type=int, default=224,
                       help='ViT classifier input resolution; fixed at 224 for ViT-L.')
    parse.set_defaults(lr=5e-5, weight_decay=0.05, adv_eval_freq=5)
    args = parse.parse_args()

    if args.input_size != 224:
        raise ValueError('This ViT pipeline is defined for input-size=224.')
    if args.model not in ('ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'):
        raise ValueError('Use --model ViT-B_16, ViT-B_32, ViT-L_16, or ViT-L_32.')
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
    info['vit_design'] = 2
    info['vit_input_size'] = 224
    logger.log(f'Using {args.model} input size: 224x224; attack space: native 32x32')

    batch_size = args.batch_size
    batch_size_validation = args.batch_size_validation
    num_adv_epochs = args.num_adv_epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f'Using device: {device}')
    if args.debug:
        num_adv_epochs = 1
    torch.backends.cudnn.benchmark = True

    seed(args.seed)
    aux = args.aux_data_filename if args.unsup_fraction > 0 else None
    train_dataset, test_dataset, eval_dataset, train_loader, test_loader, eval_loader = load_data(
        data_dir, batch_size, batch_size_validation,
        use_augmentation=args.augment, use_consistency=args.consistency,
        shuffle_train=True, aux_data_filename=aux, unsup_fraction=args.unsup_fraction,
        validation=True, num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
    )
    del train_dataset, test_dataset, eval_dataset

    seed(args.seed)
    trainer = WATrainerViT2(info, args)
    if args.vit_checkpoint and not args.resume_path:
        load_vit_checkpoint(trainer, args.vit_checkpoint, logger)

    metrics = pd.DataFrame()
    old_score = [0.0, 0.0]
    logger.log(f'Standard Accuracy-\tTest: {trainer.eval(test_loader) * 100:.2f}%.')
    trainer.init_optimizer(num_adv_epochs)

    if args.resume_path:
        start_epoch = trainer.load_model_resume(os.path.join(args.resume_path, 'state-last.pt')) + 1
    else:
        start_epoch = 1

    for epoch in range(start_epoch, num_adv_epochs + 1):
        start = time.time()
        logger.log(f'======= Epoch {epoch} =======')
        last_lr = trainer.scheduler.get_last_lr()[0] if args.scheduler else args.lr
        res = trainer.train(train_loader, epoch=epoch, adversarial=True)
        test_acc = trainer.eval(test_loader)

        logger.log(f'Loss: {res["loss"]:.4f}.\tLR: {last_lr:.2e}')
        if 'clean_acc' in res:
            logger.log(f'Standard Accuracy-\tTrain: {res["clean_acc"]*100:.2f}%.\tTest: {test_acc*100:.2f}%.')
        else:
            logger.log(f'Standard Accuracy-\tTest: {test_acc*100:.2f}%.')

        epoch_metrics = {'train_' + k: v for k, v in res.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})

        if epoch % args.adv_eval_freq == 0 or epoch == num_adv_epochs:
            test_adv = trainer.eval(test_loader, adversarial=True)
            logger.log(f'Adversarial Accuracy-\tTrain: {res["adversarial_acc"]*100:.2f}%.\tTest: {test_adv*100:.2f}%.')
            epoch_metrics['test_adversarial_acc'] = test_adv
        else:
            logger.log(f'Adversarial Accuracy-\tTrain: {res["adversarial_acc"]*100:.2f}%.')

        eval_adv = trainer.eval(eval_loader, adversarial=True)
        logger.log(f'Adversarial Accuracy-\tEval: {eval_adv*100:.2f}%.')
        epoch_metrics['eval_adversarial_acc'] = eval_adv

        if eval_adv >= old_score[1]:
            old_score = [test_acc, eval_adv]
            trainer.save_model(weights)
            logger.log(f'New best checkpoint: eval_adv={eval_adv*100:.2f}%, test_clean={test_acc*100:.2f}%')

        trainer.save_model_resume(os.path.join(log_dir, 'state-last.pt'), epoch)
        logger.log(f'Time taken: {format_time(time.time() - start)}')
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(log_dir, 'stats_adv.csv'), index=False)

    logger.log('Script Completed.')


if __name__ == '__main__':
    main()
