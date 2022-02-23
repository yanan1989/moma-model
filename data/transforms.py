import torch
from typing import Any, Callable


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/datamodule/transforms.py


class SlowFastPackPathway:
  """
  Transform for converting a video clip into a list of 2 clips with
  different temporal granualirity as needed by the SlowFast video
  models.
  For more details, refere to the paper,
  Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
  "SlowFast networks for video recognition."
  https://arxiv.org/pdf/1812.03982.pdf
  Args:
      alpha (int): Number of frames to sub-sample from the given clip
      to create the second clip.
  """

  def __init__(self, alpha: int) -> None:
    super().__init__()
    self.alpha = alpha

  def __call__(self, frames: torch.Tensor) -> list[torch.Tensor]:
    """
    Args:
        frames (tensor): frames of images sampled from the video. The
        dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
        `channel` x `num frames` x `height` x `width`.
    """
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


class ApplyTransformToKeyOnList:
  """
  Applies transform to key of dictionary input wherein input is a list
  Args:
      key (str): the dictionary key the transform is applied to
      transform (callable): the transform that is applied
  """

  def __init__(self, key: str, transform: Callable) -> None:  # pyre-ignore[24]
    self._key = key
    self._transform = transform

  def __call__(
      self, x: dict[str, list[torch.Tensor]]
  ) -> dict[str, list[torch.Tensor]]:
    x[self._key] = [self._transform(a) for a in x[self._key]]
    return x


class RepeatandConverttoList:
  """
  An utility transform that repeats each value in a
  key, value-style minibatch and replaces it with a list of values.
  Useful for performing multiple augmentations.
  An example such usecase can be found in
  `pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_mvit_16x4.yaml`
  Args:
      repead_num (int): Number of times to repeat each value.
  """

  def __init__(self, repeat_num: int) -> None:
    super().__init__()
    self.repeat_num = repeat_num

  # pyre-ignore[3]
  def __call__(self, sample_dict: dict[str, Any]) -> dict[str, list[Any]]:
    for k, v in sample_dict.items():
      sample_dict[k] = self.repeat_num*[v]
    return sample_dict
