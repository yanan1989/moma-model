from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.optim import SGD
import torchmetrics

import utils


class BaseClassifierModule(LightningModule):
  def __init__(self, data, net, cfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.data = data
    self.net = net
    self.top1 = torchmetrics.Accuracy(top_k=1)
    self.top5 = torchmetrics.Accuracy(top_k=5)

  def on_train_epoch_start(self):
    """ Needed for shuffling in distributed training
    Reference:
     - https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L96
     - https://pytorch.org/docs/master/data.html#torch.utils.data.distributed.DistributedSampler
    """
    if utils.is_ddp():
      for level in self.cfg.levels:
        self.trainer.datamodule.datasets_train[level].video_sampler.set_epoch(self.trainer.current_epoch)

  def forward(self, x):
    return self.net(x)

  def configure_optimizers(self):
    optimizer = SGD(
      self.net.parameters(),
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
    optimizer.step(closure=optimizer_closure)

    # linear warmup
    if self.trainer.global_step < self.cfg.warmup_steps:
      lr_scale = min(1.0, float(self.trainer.global_step+1)/self.cfg.warmup_steps)
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale*self.cfg.lr
