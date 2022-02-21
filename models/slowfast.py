from pytorchvideo.models import create_slowfast
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
import torchmetrics


class SlowFastModule(LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    # self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    self.model = create_slowfast()
    self.metric = torchmetrics.Accuracy()

  def on_train_epoch_start(self):
    """ Needed for distributed training
    Reference:
     - https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L96
     - https://pytorch.org/docs/master/data.html#torch.utils.data.distributed.DistributedSampler
    """
    use_ddp = self.trainer._accelerator_connector.strategy == 'ddp'
    epoch = self.trainer.current_epoch
    if use_ddp:
      self.trainer.datamodule.dataset_train.video_sampler.set_epoch(epoch)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0]
    y_hat = self.model(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('train/loss', loss, batch_size=batch_size)
    self.log('train/acc', acc, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0]
    y_hat = self.model(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('val/loss', loss, batch_size=batch_size)
    self.log('val/acc', acc, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.max_epochs, last_epoch=-1)
    return [optimizer], [scheduler]
