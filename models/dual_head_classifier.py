import torch.nn as nn
import torchmetrics

from .base_classifier import BaseClassifierModule
from .ensembler import Ensembler
import utils


class DualHeadClassifierModule(BaseClassifierModule):
  def __init__(self, cfg) -> None:
    super().__init__(cfg)
    self.ensemblers = [Ensembler(num_classes=n, gpus=cfg.gpus) for n in cfg.num_classes]
    self.metrics = nn.ModuleDict({
      'act': nn.ModuleDict({'acc1': torchmetrics.Accuracy(average='micro', top_k=1),
                            'acc5': torchmetrics.Accuracy(average='micro', top_k=5)}),
      'sact': nn.ModuleDict({'acc1': torchmetrics.Accuracy(average='micro', top_k=1),
                             'acc5': torchmetrics.Accuracy(average='micro', top_k=5)})
    })

  def training_step(self, batch, batch_idx):
    losses = []
    for i, level in enumerate(self.cfg.levels):
      x, y = batch[level]['video'], batch[level]['label']
      logits = self(x)[i]

      loss = self.get_loss(logits, y)
      losses.append(loss)
      self.log(f'train/{level}/loss', loss, prog_bar=True, batch_size=x.shape[0])

      y_hat = self.get_pred(logits)
      for name, get_stat in self.metrics[level].items():
        self.log(f'train/{level}/{name}', get_stat(y_hat, y), prog_bar=True, batch_size=x.shape[0])

    return sum(losses)

  def validation_step(self, batch, batch_idx, dataloader_idx):
    x, y = batch['video'], batch['label']
    logits = self(x)[dataloader_idx]

    loss = self.get_loss(logits, y)
    self.log(f'val/{self.cfg.levels[dataloader_idx]}/loss', loss, prog_bar=True, batch_size=x.shape[0])

    y_hat = self.get_pred(logits)
    video_ids = batch['video_index'].clone()
    self.ensemblers[dataloader_idx].ensemble_at_video_level(y_hat, y, video_ids)

  def on_validation_epoch_end(self):
    for ensembler, level in zip(self.ensemblers, self.cfg.levels):
      y_hat, y = ensembler.sync_and_aggregate_results()
      if not utils.is_ddp() or utils.get_rank() == 0:
        for name, get_stat in self.metrics[level].items():
          stat = get_stat(y_hat, y)
          self.log(f'val/{level}/{name}', stat, on_epoch=True, prog_bar=True, rank_zero_only=True)
