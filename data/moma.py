from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from momaapi import MOMA
from .transforms import get_transform
import utils


def get_paths_and_cids(moma, split, level):
  assert level in ['act', 'sact'] and split in ['train', 'val', 'test']

  if level == 'act':
    ids_act = moma.get_ids_act(split=split)
    paths_act = moma.get_paths(ids_act=ids_act)
    anns_act = moma.get_anns_act(ids_act)
    cids_act = [ann_act.cid for ann_act in anns_act]
    return paths_act, cids_act

  elif level == 'sact':
    ids_sact = moma.get_ids_sact(split=split)
    paths_sact = moma.get_paths(ids_sact=ids_sact)
    anns_sact = moma.get_anns_sact(ids_sact)
    cids_sact = [ann_sact.cid for ann_sact in anns_sact]
    return paths_sact, cids_sact

  else:
    raise ValueError


def map_cids(moma, split, level, cids):
  if level == 'act':
    return moma.map_cids(split=split, cids_act=cids)

  elif level == 'sact':
    return moma.map_cids(split=split, cids_sact=cids)

  else:
    raise ValueError


def pack_labeled_video_paths(paths, cids):
  assert len(paths) == len(cids)
  labeled_video_paths = [(path, {'label': cid}) for path, cid in zip(paths, cids)]
  return labeled_video_paths


def make_datasets(moma, level, cfg):
  # pytorch-lightning does not handle iterable datasets
  # Reference: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
  transform_train, transform_val, transform_test = get_transform(cfg)

  # get raw paths and cids
  paths_train, cids_train = get_paths_and_cids(moma, 'train', level)
  paths_val, cids_val = get_paths_and_cids(moma, 'val', level)
  paths_test, cids_test = get_paths_and_cids(moma, 'test', level)

  # make cids contiguous
  if moma.paradigm == 'few-shot':
    map_cids(moma, 'train', level, cids_train)
    map_cids(moma, 'val', level, cids_val)
    map_cids(moma, 'test', level, cids_test)

  # merge validation to training
  if not cfg.load_val:
    # offset validation set class IDs
    if moma.paradigm == 'few-shot':
      cids_val = [cid_val+moma.num_classes[f'{level}_train'] for cid_val in cids_val]

    paths_train = paths_train+paths_val
    paths_val = paths_test
    cids_train = cids_train+cids_val
    cids_val = cids_test

  labeled_video_paths_train = pack_labeled_video_paths(paths_train, cids_train)
  labeled_video_paths_val = pack_labeled_video_paths(paths_val, cids_val)
  labeled_video_paths_test = pack_labeled_video_paths(paths_test, cids_test)

  dataset_train = LabeledVideoDataset(
    labeled_video_paths=labeled_video_paths_train,
    clip_sampler=make_clip_sampler('random', cfg.T*cfg.tau/cfg.fps),
    video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
    transform=transform_train,
    decode_audio=False
  )
  dataset_val = LabeledVideoDataset(
    labeled_video_paths=labeled_video_paths_val,
    clip_sampler=make_clip_sampler('uniform', cfg.T*cfg.tau/cfg.fps),
    video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
    transform=transform_val,
    decode_audio=False
  )
  dataset_test = LabeledVideoDataset(
    labeled_video_paths=labeled_video_paths_test,
    clip_sampler=make_clip_sampler('constant_clips_per_video', cfg.T*cfg.tau/cfg.fps, cfg.num_clips, cfg.num_crops),
    video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
    transform=transform_test,
    decode_audio=False
  )

  return dataset_train, dataset_val, dataset_test


class MOMADataModule(LightningDataModule):
  def __init__(self, cfg) -> None:
    super().__init__()
    self.cfg = cfg
    self.moma = MOMA(cfg.dir_moma, paradigm=cfg.paradigm)
    self.datasets_train, self.datasets_val, self.datasets_test = {}, {}, {}

  def setup(self, stage=None):
    for level in self.cfg.levels:
      self.datasets_train[level], self.datasets_val[level], self.datasets_test[level] = \
          make_datasets(self.moma, level, self.cfg)
      print(f'Level {level}: '
            f'train={self.datasets_train[level].num_videos}, '
            f'val={self.datasets_val[level].num_videos}, '
            f'test={self.datasets_test[level].num_videos}')

  def train_dataloader(self):
    loaders = {}
    for level in self.cfg.levels:
      loaders[level] = DataLoader(self.datasets_train[level],
                                  batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                  num_workers=self.cfg.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    return loaders[self.cfg.levels[0]] if len(loaders) == 1 else loaders

  def val_dataloader(self):
    loaders = []
    for level in self.cfg.levels:
      loaders.append(DataLoader(self.datasets_val[level],
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                shuffle=False,
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=False))
    return loaders[0] if len(loaders) == 1 else loaders

  def test_dataloader(self):
    loaders = []
    for level in self.cfg.levels:
      loaders.append(DataLoader(self.datasets_test[level],
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                shuffle=False,
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=False))
    return loaders[0] if len(loaders) == 1 else loaders
