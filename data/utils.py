from pytorchvideo.data.utils import MultiProcessSampler


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py


def monkey(self) -> dict:
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
