from functools import partial
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .nets import get_net
import utils


class BaseClassifierModule(LightningModule):
  def __init__(self, cfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.net = get_net(cfg)

    self.get_loss = F.cross_entropy
    self.get_pred = partial(F.softmax, dim=-1)

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
    if self.cfg.optimizer == 'sgd':
      optimizer = SGD(self.net.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)
    elif self.cfg.optimizer == 'adamw':
      optimizer = AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)
    else:
      raise NotImplementedError
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
