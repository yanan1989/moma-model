import argparse
from pathlib import Path

from momaapi import MOMAAPI

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda, RandomCrop, CenterCrop, RandomHorizontalFlip
from pytorch_lightning import LightningModule
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler, UniformClipSampler
from pytorchvideo.transforms import ApplyTransformToKey, Normalize, RandomShortSideScale, ShortSideScale, UniformTemporalSubsample


class PackPathway(torch.nn.Module):
  """
  Transform for converting video frames as a list of tensors.
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
  def __int__(self):
    super().__init__()
    self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

  def forward(self, x):
    pass

  def training_step(self, batch, batch_idx):
    pass

  def configure_optimizers(self):
    pass


def main(cfg):
  # prepare the model
  # model = SlowFastModel()

  # prepare the datasets
  moma = MOMAAPI(cfg.dir_moma, toy=True)
  
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
        UniformTemporalSubsample(8),
        Lambda(lambda x: x/255.0),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop(244),
        RandomHorizontalFlip(p=0.5),
        PackPathway()
      ])
    ),
  ])
  transform_val = Compose([
    ApplyTransformToKey(
      key='video',
      transform=Compose([
        UniformTemporalSubsample(8),
        Lambda(lambda x: x/255.0),
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ShortSideScale(size=256),
        CenterCrop(256),
        PackPathway()
      ])
    ),
  ])

  moma_train = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_train, transform=transform_train,
                                   clip_sampler=RandomClipSampler(clip_duration=2), decode_audio=False)
  # moma_val = LabeledVideoDataset(labeled_video_paths=labeled_video_paths_val, transform=transform_val,
  #                                clip_sampler=UniformClipSampler(clip_duration=2), decode_audio=False)

  dataloader_train = DataLoader(moma_train, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
  # dataloader_val = DataLoader(moma_train, batch_size=batch_size, num_workers=num_workers)

  for i in range(4):
    for j, data in enumerate(dataloader_train):
      print(f'\nepoch {i} step {j}')
      print(data['video'][0].shape, data['video'][1].shape, data['video_name'], data['video_index'], data['clip_index'], data['aug_index'], data['label'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # computing
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--num-workers', type=int, default=4)

  # file system
  parser.add_argument('--dir-moma', type=str, default=f'{Path.home()}/data/moma')

  # hyperparameters
  parser.add_argument('--batch-size', type=int, default=4)

  args = parser.parse_args()

  main(args)
