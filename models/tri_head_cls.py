from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import LightningModule
import torch
from torch.optim import SGD
import torch.nn.functional as F
import torchmetrics


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
    batch_size = batch['act']['video'][0].shape[0] if isinstance(batch['act']['video'], list) \
                                                   else batch['act']['video'].shape[0]

    label_mask = []
    for video_name, time in zip(batch['act']['video_name'], batch['act']['time']):
      id_act = video_name.replace('.mp4', '')
      label_mask.append(self.moma.is_sact(id_act, time))
    label_mask = torch.Tensor(label_mask).type_as(batch['act']['label'])

    y_hat_act, y_hat_mask, _ = self.module(batch['act']['video'])
    _, _, y_hat_sact = self.module(batch['sact']['video'])

    loss_act = F.cross_entropy(y_hat_act, batch['act']['label'])
    loss_mask = F.binary_cross_entropy_with_logits(y_hat_mask[..., 0], label_mask.float())
    loss_sact = F.cross_entropy(y_hat_sact, batch['sact']['label'])

    pred_act = F.softmax(y_hat_act, dim=-1)
    pred_mask = torch.sigmoid(y_hat_mask[..., 0])
    pred_sact = F.softmax(y_hat_sact, dim=-1)

    top1_act = self.top1(pred_act, batch['act']['label'])
    top5_act = self.top5(pred_act, batch['act']['label'])
    acc_mask = self.acc(pred_mask, label_mask)
    top1_sact = self.top1(pred_sact, batch['sact']['label'])
    top5_sact = self.top5(pred_sact, batch['sact']['label'])

    self.log('train/act/loss', loss_act, batch_size=batch_size, prog_bar=True)
    self.log('train/mask/loss', loss_mask, batch_size=batch_size, prog_bar=True)
    self.log('train/sact/loss', loss_sact, batch_size=batch_size, prog_bar=True)
    self.log('train/act/acc1', top1_act, batch_size=batch_size, prog_bar=True)
    self.log('train/act/acc5', top5_act, batch_size=batch_size)
    self.log('train/mask/acc', acc_mask, batch_size=batch_size, prog_bar=True)
    self.log('train/sact/acc1', top1_sact, batch_size=batch_size, prog_bar=True)
    self.log('train/sact/acc5', top5_sact, batch_size=batch_size)
    return loss_act+loss_mask+loss_sact

  def validation_step(self, batch, batch_idx, dataloader_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]

    if dataloader_idx == 0:  # act
      label_mask = []
      for video_name, time in zip(batch['video_name'], batch['time']):
        id_act = video_name.replace('.mp4', '')
        label_mask.append(self.moma.is_sact(id_act, time))
      label_mask = torch.Tensor(label_mask).type_as(batch['label'])

      y_hat_act, y_hat_mask, _ = self.module(batch['video'])

      loss_act = F.cross_entropy(y_hat_act, batch['label'])
      loss_mask = F.binary_cross_entropy_with_logits(y_hat_mask[..., 0], label_mask.float())

      pred_act = F.softmax(y_hat_act, dim=-1)
      pred_mask = torch.sigmoid(y_hat_mask[..., 0])

      top1_act = self.top1(pred_act, batch['label'])
      top5_act = self.top5(pred_act, batch['label'])
      acc_mask = self.acc(pred_mask, label_mask)

      self.log('val/act/loss', loss_act, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
      self.log('val/mask/loss', loss_mask, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
      self.log('val/act/acc1', top1_act, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
      self.log('val/act/acc5', top5_act, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False)
      self.log('val/mask/acc', acc_mask, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=True)

    else:
      assert dataloader_idx == 1

      _, _, y_hat_sact = self.module(batch['video'])

      loss_sact = F.cross_entropy(y_hat_sact, batch['label'])

      pred_sact = F.softmax(y_hat_sact, dim=-1)

      top1_sact = self.top1(pred_sact, batch['label'])
      top5_sact = self.top5(pred_sact, batch['label'])

      self.log('val/sact/loss', loss_sact, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=False)
      self.log('val/sact/acc1', top1_sact, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=True)
      self.log('val/sact/acc5', top5_sact, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=False)
      # for checkpoint saving
      self.log('val/acc1', top1_sact, batch_size=batch_size, sync_dist=True, add_dataloader_idx=False, prog_bar=False)

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
