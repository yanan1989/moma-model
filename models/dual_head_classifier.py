import torch.nn.functional as F

from .base_classifier import BaseClassifierModule
from .ensembler import Ensembler
import utils


class DualHeadClassifierModule(BaseClassifierModule):
  def __init__(self, cfg) -> None:
    super().__init__(cfg)
    self.ensemblers = [Ensembler(num_classes=n, gpus=cfg.gpus) for n in cfg.num_classes]

  def training_step(self, batch, batch_idx):
    batch_size = batch['act']['video'][0].shape[0] if isinstance(batch['act']['video'], list) \
      else batch['act']['video'].shape[0]

    y_hat_act, _ = self(batch['act']['video'])
    _, y_hat_sact = self(batch['sact']['video'])

    loss_act = F.cross_entropy(y_hat_act, batch['act']['label'])
    loss_sact = F.cross_entropy(y_hat_sact, batch['sact']['label'])

    pred_act = F.softmax(y_hat_act, dim=-1)
    pred_sact = F.softmax(y_hat_sact, dim=-1)

    top1_act = self.top1(pred_act, batch['act']['label'])
    top5_act = self.top5(pred_act, batch['act']['label'])
    top1_sact = self.top1(pred_sact, batch['sact']['label'])
    top5_sact = self.top5(pred_sact, batch['sact']['label'])

    self.log('train/act/loss', loss_act, batch_size=batch_size, prog_bar=True)
    self.log('train/act/acc1', top1_act, batch_size=batch_size, prog_bar=True)
    self.log('train/act/acc5', top5_act, batch_size=batch_size)
    self.log('train/sact/loss', loss_sact, batch_size=batch_size, prog_bar=True)
    self.log('train/sact/acc1', top1_sact, batch_size=batch_size, prog_bar=True)
    self.log('train/sact/acc5', top5_sact, batch_size=batch_size)
    return loss_act+loss_sact

  def validation_step(self, batch, batch_idx, dataloader_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat_act, y_hat_sact = self(batch['video'])
    if dataloader_idx == 0:  # act
      loss_act = F.cross_entropy(y_hat_act, batch['label'])
      pred_act = F.softmax(y_hat_act, dim=-1)

      top1_act = self.top1(pred_act, batch['label'])
      top5_act = self.top5(pred_act, batch['label'])

      self.log('val/act/loss', loss_act, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
      self.log('val/act/acc1', top1_act, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
      self.log('val/act/acc5', top5_act, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

    else:
      assert dataloader_idx == 1
      loss_sact = F.cross_entropy(y_hat_sact, batch['label'])
      pred_sact = F.softmax(y_hat_sact, dim=-1)

      top1_sact = self.top1(pred_sact, batch['label'])
      top5_sact = self.top5(pred_sact, batch['label'])

      self.log('val/sact/loss', loss_sact, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
      self.log('val/sact/acc1', top1_sact, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
      self.log('val/sact/acc5', top5_sact, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)

  def on_validation_epoch_end(self):
    for ensembler, level in zip(self.ensemblers, self.cfg.levels):
      y_hat, y = ensembler.sync_and_aggregate_results()
      if utils.get_rank() == 0:
        for name, get_stat in self.metrics.items():
          stat = get_stat(y_hat, y)
          self.log(f'val/{level}/{name}', stat, on_epoch=True, prog_bar=True, rank_zero_only=True)
