from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler


from .transforms import get_transform
import utils


def get_labeled_video_paths(moma, level, split, few_shot):
  assert level in ['act', 'sact'] and split in ['train', 'val', 'test']
  split = 'test' if split == 'val' else split

  if level == 'act':
    ids_act = moma.get_ids_act(split=split)
    paths_act = moma.get_paths(ids_act=ids_act)
    anns_act = moma.get_anns_act(ids_act)
    cids_act = [ann_act.cid for ann_act in anns_act]
    if few_shot:
      cids_act = [moma.cid_to_cid_fs(cid_act, level, split) for cid_act in cids_act]
    labeled_video_paths = [(path, {'label': cid}) for path, cid in zip(paths_act, cids_act)]

  elif level == 'sact':
    ids_sact = moma.get_ids_sact(split=split)
    paths_sact = moma.get_paths(ids_sact=ids_sact)
    anns_sact = moma.get_anns_sact(ids_sact)
    cids_sact = [ann_sact.cid for ann_sact in anns_sact]
    if few_shot:
      cids_sact = [moma.cid_to_cid_fs(cid_sact, level, split) for cid_sact in cids_sact]
    labeled_video_paths = [(path, {'label': cid}) for path, cid in zip(paths_sact, cids_sact)]

  else:
    raise NotImplementedError

  return labeled_video_paths


def make_datasets(moma, level, cfg):
  # pytorch-lightning does not handle iterable datasets
  # Reference: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp

  transform_train, transform_val, transform_test = get_transform(cfg)

  dataset_train = LabeledVideoDataset(
    labeled_video_paths=get_labeled_video_paths(moma, level, 'train', cfg.few_shot),
    clip_sampler=make_clip_sampler('random', cfg.T*cfg.tau/cfg.fps),
    video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
    transform=transform_train,
    decode_audio=False
  )
  dataset_val = LabeledVideoDataset(
    labeled_video_paths=get_labeled_video_paths(moma, level, 'val', cfg.few_shot),
    clip_sampler=make_clip_sampler('uniform', cfg.T*cfg.tau/cfg.fps),
    video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
    transform=transform_val,
    decode_audio=False
  )
  dataset_test = LabeledVideoDataset(
    labeled_video_paths=get_labeled_video_paths(moma, level, 'test', cfg.few_shot),
    clip_sampler=make_clip_sampler('constant_clips_per_video', cfg.T*cfg.tau/cfg.fps, cfg.num_clips, cfg.num_crops),
    video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
    transform=transform_test,
    decode_audio=False
  )

  return dataset_train, dataset_val, dataset_test


class MOMADataModule(LightningDataModule):
  def __init__(self, moma, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg
    self.datasets_train, self.datasets_val, self.datasets_test = {}, {}, {}
    self.num_classes = {'act': 20, 'sact': 91, 'act_src': 15, 'sact_src': 69, 'act_trg': 10, 'sact_trg': 22}

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
