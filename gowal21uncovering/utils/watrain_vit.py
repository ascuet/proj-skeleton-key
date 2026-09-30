import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn

from .watrain import WATrainer, ema_update

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WATrainerViT(WATrainer):
    def init_optimizer(self, num_epochs):
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if ('norm' in n
                        or n.endswith('.bias')
                        or 'cls_token' in n
                        or 'pos_embed' in n):
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            return [
                dict(params=group_decay),
                dict(params=group_no_decay, weight_decay=0.0),
            ]

        self.optimizer = torch.optim.AdamW(
            group_weight(self.model),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)

    def train(self, dataloader, epoch=0, adversarial=False, verbose=False):
        metrics = pd.DataFrame()
        self.model.train()

        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            update_iter += 1

            x, y = data
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

            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()

            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(self.wa_model, self.model, global_step,
                       decay_rate=self.params.tau,
                       warmup_steps=self.warmup_steps,
                       dynamic_decay=True)
            metrics = pd.concat([metrics, pd.DataFrame(batch_metrics, index=[0])], ignore_index=True)

        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()

        return dict(metrics.mean())


class WATrainerViT2(WATrainerViT):
    def train(self, dataloader, epoch=0, adversarial=False, verbose=False):
        metrics = pd.DataFrame()
        self.model.train()

        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            update_iter += 1
            x, y = data
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Important: clear gradients every batch.  WATrainerViT.train omits
            # this call, which otherwise accumulates gradients across batches.
            self.optimizer.zero_grad(set_to_none=True)

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
            metrics = pd.concat(
                [metrics, pd.DataFrame(batch_metrics, index=[0])],
                ignore_index=True,
            )

        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()

        return dict(metrics.mean())
