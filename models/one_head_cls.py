from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
import torch.nn.functional as F
import torchmetrics


class OneHeadVideoClsModule(LightningModule):
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

    self.log('train/loss', loss, batch_size=batch_size, prog_bar=True)
    self.log('train/acc1', top1, batch_size=batch_size, prog_bar=True)
    self.log('train/acc5', top5, batch_size=batch_size, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    y_hat = self.module(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    pred = F.softmax(y_hat, dim=-1)
    top1 = self.top1(pred, batch['label'])
    top5 = self.top1(pred, batch['label'])

    self.log('val/loss', loss, batch_size=batch_size, sync_dist=True)
    self.log('val/acc1', top1, batch_size=batch_size, sync_dist=True, prog_bar=True)
    self.log('val/acc5', top5, batch_size=batch_size, sync_dist=True, prog_bar=True)

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
