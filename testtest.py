import argparse
from pathlib import Path

from momaapi import MOMAAPI

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision.transforms import Compose, Lambda, RandomCrop, UniformCropVideo, RandomHorizontalFlip
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler, UniformClipSampler
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, RandomShortSideScale, ShortSideScale, UniformTemporalSubsample


class PackPathway(torch.nn.Module):
  """ Transform for converting video frames as a list of tensors.
  """
  def __init__(self, alpha=4):
    super().__init__()
    self.alpha = alpha

  def forward(self, frames: torch.Tensor):
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
      frames,
      1,
      torch.linspace(
        0, frames.shape[1]-1, frames.shape[1]//self.alpha
      ).long(),
    )
    frame_list = [slow_pathway, fast_pathway]
    return frame_list


class SlowFastModel(LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    # self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    self.model = torch.hub.load('facebookresearch/pytorchvideo', 'mvit_base_16x4', pretrained=True)

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
    x = batch['video']
    y_hat = self.model(x)
    loss = F.cross_entropy(y_hat, batch['label'])
    acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch['label'])

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs, last_epoch=-1)
    return [optimizer], [scheduler]


class MOMADataModule(LightningDataModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

  def setup(self, stage=None):
    moma = MOMAAPI(self.cfg.dir_moma, toy=True)
    
    ids_sact_train = moma.get_ids_sact(split='train')
    paths_sact_train = moma.get_paths(ids_sact=ids_sact_train)
    anns_sact_train = moma.get_anns_sact(ids_sact_train)
    cids_sact_train = [ann_sact_train.cid for ann_sact_train in anns_sact_train]
    labeled_video_paths_train = [(path, {'label': cid}) for path, cid in zip(paths_sact_train, cids_sact_train)]

    ids_sact_val = moma.get_ids_sact(split='val')
    paths_sact_val = moma.get_paths(ids_sact=ids_sact_val)
    anns_sact_val = moma.get_anns_sact(ids_sact_val)
    cids_sact_val = [ann_sact_val.cid for ann_sact_val in anns_sact_val]
    labeled_video_paths_val = [(path, {'label': cid}) for path, cid in zip(paths_sact_val, cids_sact_val)]

    transform_train = Compose([
      ApplyTransformToKey(
        key='video',
        transform=Compose([
          UniformTemporalSubsample(32),
          Lambda(lambda x: x/255.0),
          Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
          RandomShortSideScale(min_size=256, max_size=320),
          RandomCrop(224),
          RandomHorizontalFlip(p=0.5),
          PackPathway()
        ])
      ),
    ])

    transform_val = Compose([
      ApplyTransformToKey(
        key='video',
        transform=Compose([
          UniformTemporalSubsample(32),
          Lambda(lambda x: x/255.0),
          Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
          ShortSideScale(size=256),
          UniformCropVideo(256),
          PackPathway()
        ])
      ),
    ])

    # pytorch-lightning does not handle iterable datasets
    # Reference: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
    use_ddp = self.trainer._accelerator_connector.strategy == 'ddp'
    dataset_train = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_train,
                                        clip_sampler=RandomClipSampler(clip_duration=2*32/30),
                                        video_sampler=DistributedSampler if use_ddp else RandomSampler,
                                        transform=transform_train,
                                        decode_audio=False)
    dataset_val = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_val,
                                      clip_sampler=UniformClipSampler(clip_duration=2*32/30),
                                      video_sampler=DistributedSampler if use_ddp else RandomSampler,
                                      transform=transform_val,
                                      decode_audio=False)

    self.dataset_train = dataset_train
    self.dataset_val = dataset_val

  def train_dataloader(self):
    dataloader_train = DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    return dataloader_train

  # def val_dataloader(self):
  #   dataloader_val = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
  #   return dataloader_val


def main(cfg):
  model = SlowFastModel(cfg)
  datamodule = MOMADataModule(cfg)

  trainer = Trainer(gpus=[0, 1], strategy='ddp')
  trainer.fit(model, datamodule)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # computing
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--num-workers', type=int, default=2)

  # file system
  parser.add_argument('--dir-moma', type=str, default=f'{Path.home()}/datamodule/moma')

  # hyperparameters
  parser.add_argument('--batch-size', type=int, default=2)
  parser.add_argument('--lr', type=float, default=1e-1)
  parser.add_argument('--num_epochs', type=int, default=200)

  args = parser.parse_args()

  main(args)
