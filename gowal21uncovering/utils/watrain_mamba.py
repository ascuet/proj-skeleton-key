import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from .trades import trades_loss, trades_loss_LSE
from .cutmix import cutmix


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainerMamba(Trainer):

    def __init__(self, info, args):
        super().__init__(info, args)

        seed(args.seed)
        self.wa_model = copy.deepcopy(self.model)

        # Match the reference WA trainer: evaluation uses 4x the training
        # attack iterations while keeping eps and step size unchanged.
        self.eval_attack = create_attack(
            self.wa_model,
            CWLoss,
            args.attack,
            args.attack_eps,
            4 * args.attack_iter,
            args.attack_step,
        )

        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples

        if self.params.data in ['cifar10', 'cifar10s', 'svhn', 'svhns']:
            self.num_classes = 10
        elif self.params.data in ['cifar100', 'cifar100s']:
            self.num_classes = 100
        elif self.params.data in ['tiny-imagenet', 'tiny-imagenets']:
            self.num_classes = 200
        else:
            raise ValueError(f'Unsupported dataset for Mamba WA trainer: {self.params.data}')

        self.update_steps = int(np.floor(num_samples / self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps

        # Persisted across save/resume.
        self.best_clean_acc = 0.0
        self.best_eval_adv_acc = 0.0

    def _parameter_groups(self):
        decay = []
        no_decay = []
        seen = set()

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in seen:
                raise RuntimeError(f'Duplicate optimizer parameter detected: {name}')
            seen.add(id(param))

            leaf = name.rsplit('.', 1)[-1]
            lowered = name.lower()

            # Standard transformer/SSM practice: no decay for biases, norm
            # parameters, 1-D scale parameters, and Mamba SSM parameters.
            if (
                param.ndim <= 1
                or leaf == 'bias'
                or 'norm' in lowered
                or leaf in {'A_log', 'D'}
            ):
                no_decay.append(param)
            else:
                decay.append(param)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable) != len(decay) + len(no_decay):
            raise RuntimeError('Optimizer parameter grouping does not cover all trainable parameters.')

        return [
            {'params': decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

    def init_optimizer(self, num_epochs):
        groups = self._parameter_groups()
        opt_choice = getattr(self.params, 'optimizer', 'adamw').lower()

        if opt_choice == 'adamw':
            self.optimizer = torch.optim.AdamW(
                groups,
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
                betas=(0.9, 0.999),
            )
        elif opt_choice == 'sgd':
            self.optimizer = torch.optim.SGD(
                groups,
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
                momentum=0.9,
                nesterov=getattr(self.params, 'nesterov', True),
            )
        else:
            raise ValueError(f'Unknown optimizer choice: {opt_choice}')

        if num_epochs <= 0:
            self.scheduler = None
            return

        warmup_epochs = min(5, max(num_epochs - 1, 0))
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(num_epochs - warmup_epochs, 1),
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(num_epochs, 1),
            )

    def train(self, dataloader, epoch=0, adversarial=False, verbose=False):
        metrics = []
        self.model.train()

        update_iter = 0
        for data in tqdm(
            dataloader,
            desc=f'Epoch {epoch}: ',
            disable=not verbose,
        ):
            global_step = (epoch - 1) * self.update_steps + update_iter

            # Match the reference BN initialization behavior.
            if global_step == 0:
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1

            x, y = data

            if self.params.consistency:
                x_aug1, x_aug2 = x[0].to(device), x[1].to(device)
                y = y.to(device)
                if self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss_consistency(
                        x_aug1, x_aug2, y, beta=self.params.beta
                    )
                else:
                    raise ValueError('Consistency training requires --beta for MambaVision.')
            else:
                if self.params.CutMix:
                    x_all = []
                    y_all = []
                    for _ in range(4):
                        x_tmp, y_tmp = x.detach(), y.detach()
                        x_tmp, y_tmp = cutmix(
                            x_tmp,
                            y_tmp,
                            alpha=1.0,
                            beta=1.0,
                            num_classes=self.num_classes,
                        )
                        x_all.append(x_tmp)
                        y_all.append(y_tmp)
                    x = torch.cat(x_all, dim=0).to(device)
                    y = torch.cat(y_all, dim=0).to(device)
                else:
                    x, y = x.to(device), y.to(device)

                if adversarial:
                    if self.params.beta is not None and self.params.mart:
                        loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                    elif self.params.beta is not None and self.params.LSE:
                        loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                    elif self.params.beta is not None:
                        loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                    else:
                        loss, batch_metrics = self.adversarial_loss(x, y)
                else:
                    loss, batch_metrics = self.standard_loss(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()

            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(
                self.wa_model,
                self.model,
                global_step,
                decay_rate=self.params.tau,
                warmup_steps=self.warmup_steps,
                dynamic_decay=True,
            )
            metrics.append(batch_metrics)

        if self.scheduler is not None:
            self.scheduler.step()

        update_bn(self.wa_model, self.model)
        return dict(pd.DataFrame(metrics).mean())

    def trades_loss(self, x, y, beta):
        loss, batch_metrics = trades_loss(
            self.model,
            x,
            y,
            self.optimizer,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            beta=beta,
            attack=self.params.attack,
            label_smoothing=self.params.ls,
            use_cutmix=self.params.CutMix,
        )
        return loss, batch_metrics

    def trades_loss_consistency(self, x_aug1, x_aug2, y, beta):
        x = torch.cat([x_aug1, x_aug2], dim=0)
        loss, batch_metrics = trades_loss(
            self.model,
            x,
            y.repeat(2),
            self.optimizer,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            beta=beta,
            attack=self.params.attack,
            label_smoothing=self.params.ls,
            use_cutmix=self.params.CutMix,
            use_consistency=True,
            cons_lambda=self.params.cons_lambda,
            cons_tem=self.params.cons_tem,
        )
        return loss, batch_metrics

    def trades_loss_LSE(self, x, y, beta):
        loss, batch_metrics = trades_loss_LSE(
            self.model,
            x,
            y,
            self.optimizer,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            beta=beta,
            attack=self.params.attack,
            label_smoothing=self.params.ls,
            clip_value=self.params.clip_value,
            use_cutmix=self.params.CutMix,
            num_classes=self.num_classes,
        )
        return loss, batch_metrics

    def eval(self, dataloader, adversarial=False):
        acc = 0.0
        self.wa_model.eval()

        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if adversarial:
                # The attack is performed on the original dataloader tensor
                # (32x32 for CIFAR). MambaVisionWrapper handles the resize
                # to args.input_size inside the model forward.
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)

                with torch.autocast(
                    device_type='cuda',
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    out = self.wa_model(x_adv)
            else:
                with torch.autocast(
                    device_type='cuda',
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    out = self.wa_model(x)

            acc += accuracy(y, out)

        if len(dataloader) == 0:
            raise RuntimeError('Cannot evaluate an empty dataloader.')
        return acc / len(dataloader)

    def save_model(self, path):
        torch.save(
            {
                'model_state_dict': self.wa_model.state_dict(),
                'unaveraged_model_state_dict': self.model.state_dict(),
            },
            path,
        )

    def save_model_resume(self, path, epoch):
        if self.scheduler is None:
            scheduler_state = None
        else:
            scheduler_state = self.scheduler.state_dict()

        torch.save(
            {
                'model_state_dict': self.wa_model.state_dict(),
                'unaveraged_model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state,
                'epoch': int(epoch),
                'best_clean_acc': float(self.best_clean_acc),
                'best_eval_adv_acc': float(self.best_eval_adv_acc),
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError(f'Model weights not found at {path}.')
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])

    def load_model_resume(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        required = [
            'model_state_dict',
            'unaveraged_model_state_dict',
            'optimizer_state_dict',
            'epoch',
        ]
        missing = [k for k in required if k not in checkpoint]
        if missing:
            raise RuntimeError(f'Resume checkpoint missing keys {missing}: {path}')

        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler_state = checkpoint.get('scheduler_state_dict')
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        elif self.scheduler is not None and scheduler_state is None:
            raise RuntimeError(f'Resume checkpoint has no scheduler state: {path}')

        self.best_clean_acc = float(checkpoint.get('best_clean_acc', 0.0))
        self.best_eval_adv_acc = float(checkpoint.get('best_eval_adv_acc', 0.0))

        return int(checkpoint['epoch'])


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor

    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean.copy_(module2.running_mean)
            module1.running_var.copy_(module2.running_var)
            module1.num_batches_tracked.copy_(module2.num_batches_tracked)
