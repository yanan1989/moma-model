from pytorch_lightning import LightningDataModule
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler, UniformClipSampler
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from .transforms import get_mvit_transforms, get_slowfast_transforms
from .utils import monkey


def get_labeled_video_paths(moma, level, split):
  assert level in ['act', 'sact'] and split in ['train', 'val']

  if level == 'act':
    ids_act = moma.get_ids_act(split=split)
    paths_act = moma.get_paths(ids_act=ids_act)
    anns_act = moma.get_anns_act(ids_act)
    cids_act = [ann_act.cid for ann_act in anns_act]
    labeled_video_paths = [(path, {'label': cid}) for path, cid in zip(paths_act, cids_act)]

  else:  # level == 'sact'
    ids_sact = moma.get_ids_sact(split=split)
    paths_sact = moma.get_paths(ids_sact=ids_sact)
    anns_sact = moma.get_anns_sact(ids_sact)
    cids_sact = [ann_sact.cid for ann_sact in anns_sact]
    labeled_video_paths = [(path, {'label': cid}) for path, cid in zip(paths_sact, cids_sact)]

  return labeled_video_paths


def make_datasets(moma, level, cfg):
  labeled_video_paths_train = get_labeled_video_paths(moma, level, 'train')
  labeled_video_paths_val = get_labeled_video_paths(moma, level, 'val')

  is_ddp = False

  # pytorch-lightning does not handle iterable datasets
  # Reference: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
  if torch.distributed.is_available() and torch.distributed.is_initialized():
    video_sampler = DistributedSampler
    is_ddp = True
  else:
    video_sampler = RandomSampler

  if cfg.backbone == 'mvit':
    transform_train, transform_val = get_mvit_transforms(cfg.mvit.T)
    clip_sampler_train = RandomClipSampler(clip_duration=cfg.mvit.T*cfg.mvit.tau/cfg.fps)
    clip_sampler_val = UniformClipSampler(clip_duration=cfg.mvit.T*cfg.mvit.tau/cfg.fps)
  else:
    assert cfg.backbone == 'slowfast'
    transform_train, transform_val = get_slowfast_transforms(cfg.slowfast.T, cfg.slowfast.alpha)
    clip_sampler_train = RandomClipSampler(clip_duration=cfg.slowfast.T*cfg.slowfast.tau/cfg.fps)
    clip_sampler_val = UniformClipSampler(clip_duration=cfg.slowfast.T*cfg.slowfast.tau/cfg.fps)

  # monkey patching
  LabeledVideoDataset.__next__ = monkey

  dataset_train = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_train,
                                      clip_sampler=clip_sampler_train,
                                      video_sampler=video_sampler,
                                      transform=transform_train,
                                      decode_audio=False)
  dataset_val = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_val,
                                    clip_sampler=clip_sampler_val,
                                    video_sampler=video_sampler,
                                    transform=transform_val,
                                    decode_audio=False)

  return dataset_train, dataset_val, is_ddp
  

class MOMAOneLevelDataModule(LightningDataModule):
  def __init__(self, moma, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg

  def setup(self, stage=None):
    self.dataset_train, self.dataset_val, is_ddp = make_datasets(self.moma, self.cfg.level, self.cfg)
    print(f'training set size: {self.dataset_train.num_videos}, '
          f'validation set size: {self.dataset_val.num_videos}')
    print(f'is_ddp: {is_ddp}')

  def train_dataloader(self):
    dataloader = DataLoader(self.dataset_train,
                            batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                            num_workers=self.cfg.num_workers,
                            pin_memory=True,
                            drop_last=True)
    return dataloader

  def val_dataloader(self):
    dataloader = DataLoader(self.dataset_val,
                            batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                            num_workers=self.cfg.num_workers,
                            pin_memory=True,
                            drop_last=False)
    return dataloader


class MOMATwoLevelDataModule(LightningDataModule):
  def __init__(self, moma, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg

  def setup(self, stage=None):
    self.dataset_act_train, self.dataset_act_val, is_ddp_act = make_datasets(self.moma, 'act', self.cfg)
    self.dataset_sact_train, self.dataset_sact_val, is_ddp_sact = make_datasets(self.moma, 'sact', self.cfg)
    print(f'training set size: [act={self.dataset_act_train.num_videos}, '
          f'sact={self.dataset_sact_train.num_videos}], '
          f'validation set size: [act={self.dataset_act_val.num_videos}, '
          f'sact={self.dataset_sact_val.num_videos}]')
    print(f'is_ddp: {is_ddp_act and is_ddp_sact}')

  def train_dataloader(self):
    dataloader_act = DataLoader(self.dataset_act_train,
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=True)
    dataloader_sact = DataLoader(self.dataset_sact_train,
                                 batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    return {'act': dataloader_act, 'sact': dataloader_sact}

  def val_dataloader(self):
    dataloader_act = DataLoader(self.dataset_act_val,
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=False)
    dataloader_sact = DataLoader(self.dataset_sact_val,
                                 batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

    return [dataloader_act, dataloader_sact]


def get_data(moma, cfg):
  if cfg.level in ['act', 'sact']:
    return MOMAOneLevelDataModule(moma, cfg)
  else:
    assert cfg.level == 'both'
    return MOMATwoLevelDataModule(moma, cfg)
