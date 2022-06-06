import itertools
import numpy as np
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from torch.utils.data import DistributedSampler, RandomSampler

import utils


class MOMAActDataset(LabeledVideoDataset):
  def __init__(self, cfg, moma, split, transform):
    if isinstance(split, list):
      ids_act = list(itertools.chain(moma.get_ids_act(split=x) for x in split))
    else:
      ids_act = moma.get_ids_act(split=split)

    paths_act = moma.get_paths(ids_act=ids_act)
    anns_act = moma.get_anns_act(ids_act)
    cids_act = [ann_act.cid for ann_act in anns_act]

    # make cids contiguous
    if moma.paradigm == 'few-shot':
      cids_act = moma.map_cids(split=split, cids_act=cids_act)

    labeled_video_paths = [(path_act, {'cid_act': cid_act}) for path_act, cid_act in zip(paths_act, cids_act)]

    if split == 'train' or 'train' in split:
      clip_sampler = make_clip_sampler('random', cfg.T*cfg.tau/cfg.fps)
    elif split == 'val':
      clip_sampler = make_clip_sampler('uniform', cfg.T*cfg.tau/cfg.fps)
    elif split == 'test':
      clip_sampler = make_clip_sampler('constant_clips_per_video', cfg.T*cfg.tau/cfg.fps, cfg.num_clips, cfg.num_crops)
    else:
      raise ValueError

    super().__init__(
      labeled_video_paths=labeled_video_paths,
      clip_sampler=clip_sampler,
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform,
      decode_audio=False
    )


class MOMASActDataset(LabeledVideoDataset):
  def __init__(self, cfg, moma, split, transform):
    ids_sact = moma.get_ids_sact(split=split)

    paths_sact = moma.get_paths(ids_sact=ids_sact)
    anns_sact = moma.get_anns_sact(ids_sact)
    cids_sact = [ann_sact.cid for ann_sact in anns_sact]

    ids_act = moma.get_ids_act(ids_sact=ids_sact)
    anns_act = moma.get_anns_act(ids_act)
    cids_act = [ann_act.cid for ann_act in anns_act]

    # make cids contiguous
    if moma.paradigm == 'few-shot':
      cids_sact = moma.map_cids(split=split, cids_sact=cids_sact)
      cids_act = moma.map_cids(split=split, cids_act=cids_act)

    labeled_video_paths = [(path_sact, {'cid_sact': cid_sact, 'cid_act': cid_act})
                           for path_sact, cid_sact, cid_act in zip(paths_sact, cids_sact, cids_act)]

    if split == 'train':
      clip_sampler = make_clip_sampler('random', cfg.T*cfg.tau/cfg.fps)
    elif split == 'val':
      clip_sampler = make_clip_sampler('uniform', cfg.T*cfg.tau/cfg.fps)
    elif split == 'test':
      clip_sampler = make_clip_sampler('constant_clips_per_video', cfg.T*cfg.tau/cfg.fps, cfg.num_clips, cfg.num_crops)
    else:
      raise ValueError

    super().__init__(
      labeled_video_paths=labeled_video_paths,
      clip_sampler=clip_sampler,
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform,
      decode_audio=False
    )


class MOMAAActDataset(LabeledVideoDataset):
  def __init__(self, cfg, moma, split, transform):
    ids_sact = moma.get_ids_sact(split=split)
    paths_sact = moma.get_paths(ids_sact=ids_sact)
    anns_sact = moma.get_anns_sact(ids_sact)
    cids_sact = [ann_sact.cid for ann_sact in anns_sact]
    cids_aact = np.stack([aact_actor.cids_predicate for ann_sact in anns_sact for aact_actor in ann_sact.aacts_actor])

    ids_act = moma.get_ids_act(ids_sact=ids_sact)
    anns_act = moma.get_anns_act(ids_act)
    cids_act = [ann_act.cid for ann_act in anns_act]

    # make cids contiguous
    if moma.paradigm == 'few-shot':
      cids_sact = moma.map_cids(split=split, cids_sact=cids_sact)
      cids_act = moma.map_cids(split=split, cids_act=cids_act)

    labeled_video_paths = [(path_sact, {'cid_sact': cid_sact, 'cid_act': cid_act, 'cids_aact': cids_aact})
                           for path_sact, cid_sact, cid_act in zip(paths_sact, cids_sact, cids_act)]

    if split == 'train':
      clip_sampler = make_clip_sampler('random', cfg.T*cfg.tau/cfg.fps)
    elif split == 'val':
      clip_sampler = make_clip_sampler('uniform', cfg.T*cfg.tau/cfg.fps)
    elif split == 'test':
      clip_sampler = make_clip_sampler('constant_clips_per_video', cfg.T*cfg.tau/cfg.fps, cfg.num_clips, cfg.num_crops)
    else:
      raise ValueError

    super().__init__(
      labeled_video_paths=labeled_video_paths,
      clip_sampler=clip_sampler,
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform,
      decode_audio=False
    )