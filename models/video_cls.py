import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
import torch.nn.functional as F
import torchmetrics

from .backbone import get_mvit_backbone, get_slowfast_backbone


def get_module(cfg):
  if cfg.level == 'act':
    num_classes = cfg.num_classes.act
  elif cfg.level == 'sact':
    num_classes = cfg.num_classes.sact
  else:
    assert cfg.level == 'both'
    num_classes = (cfg.num_classes.act, cfg.num_classes.sct)

  if cfg.backbone == 'mvit':
    module = get_mvit_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')
  else:
    assert cfg.backbone == 'slowfast'
    module = get_slowfast_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')
  return module


class VideoClassificationModule(LightningModule):
  def __init__(self, moma, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg
    self.module = get_module(cfg)
    self.metric = torchmetrics.Accuracy()

  def on_train_epoch_start(self):
    """ Needed for shuffling in distributed training
    Reference:
     - https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L96
     - https://pytorch.org/docs/master/data.html#torch.utils.data.distributed.DistributedSampler
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      self.trainer.datamodule.dataset_train.video_sampler.set_epoch(self.trainer.current_epoch)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    is_sact = []
    for video_name, time in zip(batch['video_name'], batch['clip_index']):
      id_act = video_name.replace('.mp4', '')
      is_sact.append(self.moma.is_sact(id_act, time))
    is_sact = torch.Tensor(is_sact).type_as(batch['label'])

    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat = self.module(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('train/loss', loss, batch_size=batch_size)
    self.log('train/acc', acc, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat = self.module(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('val/loss', loss, batch_size=batch_size)
    self.log('val/acc', acc, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = SGD(
      self.parameters(),
      lr=self.cfg.lr,
      momentum=self.cfg.momentum,
      weight_decay=self.cfg.wd
    )
    scheduler = CosineAnnealingLR(optimizer, self.cfg.num_epochs)

    return [optimizer], [scheduler]

  def optimizer_step(
      self,
      epoch,
      batch_idx,
      optimizer,
      optimizer_idx,
      optimizer_closure,
      on_tpu=False,
      using_native_amp=False,
      using_lbfgs=False
  ):
    # linear warmup
    if self.trainer.global_step < self.cfg.warmup_steps:
      lr_scale = min(1.0, float(self.trainer.global_step+1)/self.cfg.warmup_steps)
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale*self.cfg.lr

    optimizer.step(closure=optimizer_closure)


def get_model(moma, cfg):
  return VideoClassificationModule(moma, cfg)
