from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
import torchmetrics


class BaseVideoClsModule(LightningModule):
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
      for level in self.cfg.levels:
        self.trainer.datamodule.datasets_train[level].video_sampler.set_epoch(self.trainer.current_epoch)

  def forward(self, x):
    return self.module(x)

  def configure_optimizers(self):
    optimizer = SGD(
      self.module.parameters(),
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
