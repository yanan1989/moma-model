from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torchmetrics


class VideoClassificationModule(LightningModule):
  def __init__(self, optim: DictConfig, backbone: DictConfig) -> None:
    super().__init__()
    self.optim = optim
    self.backbone = instantiate(backbone)
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
    y_hat = self.backbone(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('train/loss', loss, batch_size=batch_size)
    self.log('train/acc', acc, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0]
    y_hat = self.backbone(batch['video'])
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.metric(F.softmax(y_hat, dim=-1), batch['label'])

    self.log('val/loss', loss, batch_size=batch_size)
    self.log('val/acc', acc, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = instantiate(self.optim.optimizer, params=self.parameters())
    scheduler = instantiate(self.optim.scheduler, optimizer=optimizer)
    return [optimizer], [scheduler]
