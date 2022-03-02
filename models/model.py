from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
import torch.nn.functional as F
import torchmetrics

from .backbone import get_mvit_backbone, get_slowfast_backbone


class VideoClsModule(LightningModule):
  def __init__(self, moma, backbone, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg
    self.module = backbone
    self.top1 = torchmetrics.Accuracy(top_k=1)
    self.top5 = torchmetrics.Accuracy(top_k=5)

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
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat = self.module(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    pred = F.softmax(y_hat, dim=-1)
    top1 = self.top1(pred, batch['label'])
    top5 = self.top1(pred, batch['label'])

    self.log('train/loss', loss, batch_size=batch_size)
    self.log('train/top1', top1, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    self.log('train/top5', top5, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat = self.module(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    pred = F.softmax(y_hat, dim=-1)
    top1 = self.top1(pred, batch['label'])
    top5 = self.top1(pred, batch['label'])

    self.log('val/loss', loss, batch_size=batch_size)
    self.log('val/top1', top1, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
    self.log('val/top5', top5, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
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


class TriHeadVideoClsModule(LightningModule):
  def __init__(self, moma, backbone, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg
    self.module = backbone
    self.acc = torchmetrics.Accuracy()
    self.top1 = torchmetrics.Accuracy(top_k=1)
    self.top5 = torchmetrics.Accuracy(top_k=5)

  def on_train_epoch_start(self):
    """ Needed for shuffling in distributed training
    Reference:
     - https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L96
     - https://pytorch.org/docs/master/data.html#torch.utils.data.distributed.DistributedSampler
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      self.trainer.datamodule.dataset_act_train.video_sampler.set_epoch(self.trainer.current_epoch)
      self.trainer.datamodule.dataset_sact_train.video_sampler.set_epoch(self.trainer.current_epoch)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    batch_size = batch[0]['video'][0].shape[0] if isinstance(batch[0]['video'], list) else batch[0]['video'].shape[0]

    label_mask = []
    for video_name, time in zip(batch[0]['video_name'], batch[0]['clip_index']):
      id_act = video_name.replace('.mp4', '')
      label_mask.append(self.moma.is_sact(id_act, time))
    label_mask = torch.Tensor(label_mask).type_as(batch[0]['label'])

    y_hat_act, y_hat_mask, _ = self.module(batch[0]['video'])
    _, _, y_hat_sact = self.module(batch[1]['video'])

    loss_act = F.cross_entropy(y_hat_act, batch[0]['label'])
    loss_mask = F.binary_cross_entropy_with_logits(y_hat_mask[..., 0], label_mask.float())*2
    loss_sact = F.cross_entropy(y_hat_sact, batch[1]['label'])

    pred_act = F.softmax(y_hat_act, dim=-1)
    pred_mask = torch.sigmoid(y_hat_mask[..., 0])
    pred_sact = F.softmax(y_hat_sact, dim=-1)

    top1_act = self.top1(pred_act, batch[0]['label'])
    top5_act = self.top5(pred_act, batch[0]['label'])
    top1_sact = self.top1(pred_sact, batch[1]['label'])
    top5_sact = self.top5(pred_sact, batch[1]['label'])
    acc_mask = self.acc(pred_mask, label_mask)

    self.log('train/loss_act', loss_act, batch_size=batch_size, prog_bar=True)
    self.log('train/loss_mask', loss_mask, batch_size=batch_size, prog_bar=True)
    self.log('train/loss_sact', loss_sact, batch_size=batch_size, prog_bar=True)
    self.log('train/top1_act', top1_act, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    self.log('train/top5_act', top5_act, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    self.log('train/top1_sact', top1_sact, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    self.log('train/top5_sact', top5_sact, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    self.log('train/acc_mask', acc_mask, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=False)
    return loss_act+loss_mask+loss_sact

  def validation_step(self, batch, batch_idx):
    batch_size = batch[0]['video'][0].shape[0] if isinstance(batch[0]['video'], list) else batch[0]['video'].shape[0]

    label_mask = []
    for video_name, time in zip(batch[0]['video_name'], batch[0]['clip_index']):
      id_act = video_name.replace('.mp4', '')
      label_mask.append(self.moma.is_sact(id_act, time))
    label_mask = torch.Tensor(label_mask).type_as(batch[0]['label'])

    y_hat_act, y_hat_mask, _ = self.module(batch[0]['video'])
    _, _, y_hat_sact = self.module(batch[1]['video'])

    loss_act = F.cross_entropy(y_hat_act, batch[0]['label'])
    loss_mask = F.binary_cross_entropy_with_logits(y_hat_mask[..., 0], label_mask.float())*2
    loss_sact = F.cross_entropy(y_hat_sact, batch[1]['label'])

    pred_act = F.softmax(y_hat_act, dim=-1)
    pred_mask = torch.sigmoid(y_hat_mask[..., 0])
    pred_sact = F.softmax(y_hat_sact, dim=-1)

    top1_act = self.top1(pred_act, batch[0]['label'])
    top5_act = self.top5(pred_act, batch[0]['label'])
    top1_sact = self.top1(pred_sact, batch[1]['label'])
    top5_sact = self.top5(pred_sact, batch[1]['label'])
    acc_mask = self.acc(pred_mask, label_mask)

    self.log('val/loss_act', loss_act, batch_size=batch_size, prog_bar=True)
    self.log('val/loss_mask', loss_mask, batch_size=batch_size, prog_bar=True)
    self.log('val/loss_sact', loss_sact, batch_size=batch_size, prog_bar=True)
    self.log('val/top1_act', top1_act, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
    self.log('val/top5_act', top5_act, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
    self.log('val/top1_sact', top1_sact, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
    self.log('val/top5_sact', top5_sact, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
    self.log('val/acc_mask', acc_mask, batch_size=batch_size, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss_act+loss_mask+loss_sact

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
  if cfg.level == 'act':
    num_classes = cfg.num_classes_act
  elif cfg.level == 'sact':
    num_classes = cfg.num_classes_sact
  else:
    assert cfg.level == 'both'
    num_classes = (cfg.num_classes_act, cfg.num_classes_sact)

  if cfg.backbone == 'mvit':
    backbone = get_mvit_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')
  else:
    assert cfg.backbone == 'slowfast'
    backbone = get_slowfast_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')

  if cfg.level == 'act' or cfg.level == 'sact':
    return VideoClsModule(moma, backbone, cfg)
  else:
    assert cfg.level == 'both'
    return TriHeadVideoClsModule(moma, backbone, cfg)
