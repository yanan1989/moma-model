from hydra.utils import instantiate
from omegaconf import DictConfig
import os
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
import torch.nn.functional as F
import torchmetrics


def get_module(cfg):
  module = instantiate(cfg.module)
  if cfg.strategy == 'finetune':
    weights = torch.load(os.path.join(cfg.dir_pretrain, cfg.name_pretrain))['model_state']
    weights.pop('blocks.6.proj.weight')
    weights.pop('blocks.6.proj.bias')
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
    """ Needed for distributed training
    Reference:
     - https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L96
     - https://pytorch.org/docs/master/data.html#torch.utils.data.distributed.DistributedSampler
    """
    use_ddp = self.trainer._accelerator_connector.strategy == 'ddp'
    if use_ddp:
      epoch = self.trainer.current_epoch
      self.trainer.datamodule.dataset_train.video_sampler.set_epoch(epoch)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0]
    y_hat = self.module(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('train/loss', loss, batch_size=batch_size)
    self.log('train/acc', acc, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0]
    y_hat = self.module(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('val/loss', loss, batch_size=batch_size)
    self.log('val/acc', acc, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = SGD(
      self.parameters(),
      lr=self.cfg.lr,
      momentum=self.cfg.momentum,
      weight_decay=self.cfg.wd
    )
    scheduler = LinearWarmupCosineAnnealingLR(
      optimizer,
      warmup_epochs=self.cfg.warmup_epochs,
      max_epochs=self.cfg.max_epochs,
      warmup_start_lr=self.cfg.warmup_start_lr
    )
    return [optimizer], [scheduler]
