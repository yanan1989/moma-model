import torch.nn as nn
import torchmetrics

from .base_classifier import BaseClassifierModule
from .ensembler import Ensembler
import utils


class OneHeadClassifierModule(BaseClassifierModule):
  def __init__(self, cfg) -> None:
    super().__init__(cfg)
    self.ensembler = Ensembler(num_classes=cfg.num_classes[0], gpus=cfg.gpus)
    self.metrics = nn.ModuleDict({'acc1': torchmetrics.Accuracy(average='micro', top_k=1),
                                  'acc5': torchmetrics.Accuracy(average='micro', top_k=5)})

  def training_step(self, batch, batch_idx):
    x, y = batch['video'], batch['label']
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('train/loss', loss, prog_bar=True, batch_size=x.shape[0])

    y_hat = self.get_pred(logits)
    for name, get_stat in self.metrics.items():
      self.log(f'train/{name}', get_stat(y_hat, y), prog_bar=True, batch_size=x.shape[0])

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch['video'], batch['label']
    logits = self(x)

    loss = self.get_loss(logits, y)
    self.log('val/loss', loss, sync_dist=True, prog_bar=True, batch_size=x.shape[0])

    y_hat = self.get_pred(logits)
    video_ids = batch['video_index'].clone()
    self.ensembler.ensemble_at_video_level(y_hat, y, video_ids)

  def on_validation_epoch_end(self):
    y_hat, y = self.ensembler.sync_and_aggregate_results()
    if not utils.is_ddp() or utils.get_rank() == 0:
      for name, get_stat in self.metrics.items():
        stat = get_stat(y_hat, y)
        self.log(f'val/{name}', stat, on_epoch=True, prog_bar=True, rank_zero_only=True)
