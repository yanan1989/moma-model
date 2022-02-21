from momaapi import MOMA

from pytorch_lightning import LightningDataModule
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler, UniformClipSampler
from pytorchvideo.transforms import (
  ApplyTransformToKey,
  Div255,
  Normalize,
  RandomShortSideScale,
  ShortSideScale,
  UniformTemporalSubsample
)
from torch.utils.data import DistributedSampler, RandomSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)

from .transforms import SlowFastPackPathway
from .utils import get_labeled_video_paths


class MOMADataModule(LightningDataModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg

  def setup(self, stage=None):
    moma = MOMA(self.cfg.dir_dataset, toy=True)

    labeled_video_paths_train = get_labeled_video_paths(moma, 'sact', 'train')
    labeled_video_paths_val = get_labeled_video_paths(moma, 'sact', 'val')

    clip_sampler_train = RandomClipSampler(clip_duration=self.cfg.T*self.cfg.tau/self.cfg.fps)
    clip_sampler_val = UniformClipSampler(clip_duration=self.cfg.T*self.cfg.tau/self.cfg.fps)

    # pytorch-lightning does not handle iterable datasets
    # Reference: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
    # use_ddp = self.trainer._accelerator_connector.strategy == 'ddp'
    use_ddp = False
    video_sampler_train = DistributedSampler if use_ddp else RandomSampler
    video_sampler_val = DistributedSampler if use_ddp else RandomSampler

    transform_train = Compose([
      ApplyTransformToKey(
        key='video',
        transform=Compose([
          UniformTemporalSubsample(self.cfg.T*self.cfg.alpha),
          Div255(),
          Normalize(self.cfg.mean, self.cfg.std),
          RandomShortSideScale(min_size=self.cfg.train.size_scale[0], max_size=self.cfg.train.size_scale[1]),
          RandomCrop(self.cfg.train.size_crop),
          RandomHorizontalFlip(p=0.5),
          SlowFastPackPathway(self.cfg.alpha)
        ])
      ),
    ])

    transform_val = Compose([
      ApplyTransformToKey(
        key='video',
        transform=Compose([
          UniformTemporalSubsample(self.cfg.T*self.cfg.alpha),
          Div255(),
          Normalize(self.cfg.mean, self.cfg.std),
          ShortSideScale(size=self.cfg.train.size_scale),
          CenterCrop(self.cfg.train.size_crop),
          SlowFastPackPathway(self.cfg.alpha)
        ])
      ),
    ])

    dataset_train = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_train,
                                        clip_sampler=clip_sampler_train,
                                        video_sampler=video_sampler_train,
                                        transform=transform_train,
                                        decode_audio=False)
    dataset_val = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_val,
                                      clip_sampler=clip_sampler_val,
                                      video_sampler=video_sampler_val,
                                      transform=transform_val,
                                      decode_audio=False)

    self.dataset_train = dataset_train
    self.dataset_val = dataset_val

  def train_dataloader(self):
    # dataloader_train = DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    # return dataloader_train
    pass

  def val_dataloader(self):
    # dataloader_val = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    # return dataloader_val
    pass
