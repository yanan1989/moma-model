import torch


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/datamodule/transforms.py
class SlowFastPackPathway:
  """
  Transform for converting a video clip into a list of 2 clips with
  different temporal granualirity as needed by the SlowFast video
  model.
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
