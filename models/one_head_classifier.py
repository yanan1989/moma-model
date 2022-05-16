import torch.nn.functional as F

from .base_classifier import BaseClassifierModule


class OneHeadClassifierModule(BaseClassifierModule):
  def __init__(self, data, net, cfg) -> None:
    super().__init__(data, net, cfg)

  def training_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat = self(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    pred = F.softmax(y_hat, dim=-1)
    top1 = self.top1(pred, batch['label'])
    top5 = self.top1(pred, batch['label'])

    self.log('train/loss', loss, batch_size=batch_size, prog_bar=True)
    self.log('train/acc1', top1, batch_size=batch_size, prog_bar=True)
    self.log('train/acc5', top5, batch_size=batch_size, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat = self(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    pred = F.softmax(y_hat, dim=-1)
    top1 = self.top1(pred, batch['label'])
    top5 = self.top1(pred, batch['label'])

    self.log('val/loss', loss, batch_size=batch_size, sync_dist=True, prog_bar=True)
    self.log('val/acc1', top1, batch_size=batch_size, sync_dist=True, prog_bar=True)
    self.log('val/acc5', top5, batch_size=batch_size, sync_dist=True)

  def test_step(self, batch, batch_idx, dataloader_idx):
    pass
