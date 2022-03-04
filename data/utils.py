from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler, UniformClipSampler
from pytorchvideo.data.utils import MultiProcessSampler
import torch
from torch.utils.data import DistributedSampler, RandomSampler

from .transforms import get_mvit_transforms, get_slowfast_transforms


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py
def __next__(self) -> dict:
  """
  Retrieves the next clip based on the clip sampling strategy and video sampler.
  Returns:
      A dictionary with the following format.
      .. code-block:: text
          {
              'video': <video_tensor>,
              'label': <index_label>,
              'video_label': <index_label>
              'video_index': <video_index>,
              'clip_index': <clip_index>,
              'aug_index': <aug_index>,
          }
  """
  if not self._video_sampler_iter:
    # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
    self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

  for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
    # Reuse previously stored video if there are still clips to be sampled from
    # the last loaded video.
    if self._loaded_video_label:
      video, info_dict, video_index = self._loaded_video_label
    else:
      video_index = next(self._video_sampler_iter)
      try:
        video_path, info_dict = self._labeled_videos[video_index]
        video = self.video_path_handler.video_from_path(
          video_path,
          decode_audio=self._decode_audio,
          decoder=self._decoder,
        )
        self._loaded_video_label = (video, info_dict, video_index)
      except Exception as e:
        logger.debug(
          "Failed to load video with error: {}; trial {}".format(
            e,
            i_try,
          )
        )
        continue

    (
      clip_start,
      clip_end,
      clip_index,
      aug_index,
      is_last_clip,
    ) = self._clip_sampler(
      self._next_clip_start_time, video.duration, info_dict
    )

    if isinstance(clip_start, list):  # multi-clip in each sample

      # Only load the clips once and reuse previously stored clips if there are multiple
      # views for augmentations to perform on the same clips.
      if aug_index[0] == 0:
        self._loaded_clip = {}
        loaded_clip_list = []
        for i in range(len(clip_start)):
          clip_dict = video.get_clip(clip_start[i], clip_end[i])
          if clip_dict is None or clip_dict["video"] is None:
            self._loaded_clip = None
            break
          loaded_clip_list.append(clip_dict)

        if self._loaded_clip is not None:
          for key in loaded_clip_list[0].keys():
            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

      time = [float(s+e)/2 for s, e in zip(clip_start, clip_end)]

    else:  # single clip case

      # Only load the clip once and reuse previously stored clip if there are multiple
      # views for augmentations to perform on the same clip.
      if aug_index == 0:
        self._loaded_clip = video.get_clip(clip_start, clip_end)

      time = float(clip_start+clip_end)/2

    self._next_clip_start_time = clip_end

    video_is_null = (
        self._loaded_clip is None or self._loaded_clip["video"] is None
    )
    if (
        is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
    ) or video_is_null:
      # Close the loaded encoded video and reset the last sampled clip time ready
      # to sample a new video on the next iteration.
      self._loaded_video_label[0].close()
      self._loaded_video_label = None
      self._next_clip_start_time = 0.0
      self._clip_sampler.reset()
      if video_is_null:
        logger.debug(
          "Failed to load clip {}; trial {}".format(video.name, i_try)
        )
        continue

    frames = self._loaded_clip["video"]
    audio_samples = self._loaded_clip["audio"]
    sample_dict = {
      "video": frames,
      "video_name": video.name,
      "video_index": video_index,
      "clip_index": clip_index,
      "aug_index": aug_index,
      "time": time,
      **info_dict,
      **({"audio": audio_samples} if audio_samples is not None else {}),
    }
    if self._transform is not None:
      sample_dict = self._transform(sample_dict)

      # User can force dataset to continue by returning None in transform.
      if sample_dict is None:
        continue

    return sample_dict
  else:
    raise RuntimeError(
      f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
    )


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
  LabeledVideoDataset.__next__ = __next__

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
