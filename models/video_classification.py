from hydra.utils import instantiate
from omegaconf import DictConfig
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
import torch.nn.functional as F
import torchmetrics


def get_module(cfg):
  module = instantiate(cfg.module, _convert_='all')
  if cfg.strategy == 'finetune':
    weights = torch.load(os.path.join(cfg.dir_pretrain, cfg.name_pretrain))['model_state']
    weights.pop('blocks.6.proj.weight', None)
    weights.pop('blocks.6.proj.bias', None)
    weights.pop('head.proj.weight', None)
    weights.pop('head.proj.bias', None)
    print(f'{list(set(module.state_dict())-set(weights.keys()))} will be trained from scratch')
    module.load_state_dict(weights, strict=False)
  return module


class VideoClassificationModule(LightningModule):
  def __init__(self, cfg: DictConfig) -> None:
    super().__init__()
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
      using_lbfgs=False,
  ):
    # linear warmup
    if self.trainer.global_step < self.cfg.warmup_steps:
      lr_scale = min(1.0, float(self.trainer.global_step+1)/self.cfg.warmup_steps)
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale*self.cfg.lr

    optimizer.step(closure=optimizer_closure)
