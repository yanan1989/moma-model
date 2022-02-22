from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler, UniformClipSampler
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from momaapi import MOMA


def get_labeled_video_paths(moma, level, split):
  assert level in ['act', 'sact'] and split in ['train', 'val']

  if split == 'act':
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


class MOMADataModule(LightningDataModule):
  def __init__(self, cfg: DictConfig) -> None:
    super().__init__()
    self.cfg = cfg

  def setup(self, stage=None):
    moma = MOMA(self.cfg.dir_moma)
    labeled_video_paths_train = get_labeled_video_paths(moma, self.cfg.level, 'train')
    labeled_video_paths_val = get_labeled_video_paths(moma, self.cfg.level, 'val')

    clip_sampler_train = RandomClipSampler(clip_duration=self.cfg.T*self.cfg.tau/self.cfg.fps)
    clip_sampler_val = UniformClipSampler(clip_duration=self.cfg.T*self.cfg.tau/self.cfg.fps)

    # pytorch-lightning does not handle iterable datasets
    # Reference: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
    use_ddp = self.trainer._accelerator_connector.strategy == 'ddp'
    video_sampler_train = DistributedSampler if use_ddp else RandomSampler
    video_sampler_val = DistributedSampler if use_ddp else RandomSampler

    transform_train = instantiate(self.cfg.transform.train)
    transform_val = instantiate(self.cfg.transform.val)

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
    print(f'training set size: {dataset_train.num_videos}, validation set size: {dataset_val.num_videos}')

  def train_dataloader(self):
    dataloader = DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    return dataloader

  def val_dataloader(self):
    dataloader = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)
    return dataloader
