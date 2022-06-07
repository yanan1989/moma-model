import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop
from pytorchvideo.transforms.functional import (
  clip_boxes_to_image,
  crop_boxes,
  _get_param_spatial_crop,
  horizontal_flip_with_boxes,
  random_crop_with_boxes,
  random_short_side_scale_with_boxes,
  short_side_scale_with_boxes,
  uniform_crop_with_boxes
)

"""
 - images: torch.Tensor of size num_channels x num_images x height x width
 - bboxes: torch.Tensor of size num_boxes x 4 in x1, y1, x2, y2
"""


class CenterCropWithBoxes(nn.Module):
  def __init__(self, size):
    super().__init__()
    self._size = size

  def forward(self, images, boxes):
    height, width = images.shape[2:4]
    x_offset = int(math.ceil((width-self._size)/2))
    y_offset = int(math.ceil((height-self._size)/2))

    images = center_crop(images, self._size)
    boxes = crop_boxes(boxes, x_offset, y_offset)
    boxes = clip_boxes_to_image(boxes, images.shape[-2], images.shape[-1])
    return images, boxes


class RandomCropWithBoxes(nn.Module):
  def __init__(self, size):
    super().__init__()
    self._size = size

  def forward(self, images, boxes):
    images, boxes = random_crop_with_boxes(images, self._size, boxes)
    return images, boxes


class RandomHorizontalFlipWithBoxes(nn.Module):
  def __init__(self, p=0.5):
    super().__init__()
    self._p = p

  def forward(self, images, boxes):
    images, boxes = horizontal_flip_with_boxes(self._p, images, boxes)
    return images, boxes


class RandomResizedCropWithBoxes(nn.Module):
  def __init__(self, target_height, target_width, scale, aspect_ratio, log_uniform_ratio=True, interpolation='bilinear',
               num_tries=10):
    super().__init__()
    self._target_height = target_height
    self._target_width = target_width
    self._scale = scale
    self._aspect_ratio = aspect_ratio
    self._log_uniform_ratio = log_uniform_ratio
    self._interpolation = interpolation
    self._num_tries = num_tries

  def forward(self, images, boxes):
    assert (self._scale[0] > 0 and self._scale[1] > 0), \
      'min and max of self._scale range must be greater than 0'
    assert (self._aspect_ratio[0] > 0 and self._aspect_ratio[1] > 0), \
      'min and max of self._aspect_ratio range must be greater than 0'

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(
      self._scale, self._aspect_ratio, height, width, self._log_uniform_ratio, self._num_tries
    )  # y1, x1, h, w

    images = images[:, :, i:i+h, j:j+w]
    images = F.interpolate(images, size=(self._target_height, self._target_width), mode=self._interpolation)

    boxes = crop_boxes(boxes, j, i)
    boxes[:, [0, 2]] *= float(self._target_width)/w
    boxes[:, [1, 3]] *= float(self._target_height)/h
    boxes = clip_boxes_to_image(boxes, self._target_height, self._target_width)

    return images, boxes


class RandomShortSideScaleWithBoxes(nn.Module):
  def __init__(self, min_size, max_size, interpolation='bilinear', backend='pytorch'):
    super().__init__()
    self._min_size = min_size
    self._max_size = max_size
    self._interpolation = interpolation
    self._backend = backend

  def forward(self, images, boxes):
    images, boxes = random_short_side_scale_with_boxes(images, boxes, self._min_size, self._max_size,
                                                       self._interpolation, self._backend)
    return images, boxes


class ShortSideScaleWithBoxes(nn.Module):
  def __init__(self, size, interpolation='bilinear', backend='pytorch'):
    super().__init__()
    self._size = size
    self._interpolation = interpolation
    self._backend = backend

  def forward(self, images, boxes):
    images, boxes = short_side_scale_with_boxes(images, boxes, self._size, self._interpolation, self._backend)
    return images, boxes


# TODO
class UniformCropWithBoxes(nn.Module):
  def __init__(self, size, video_key='video', aug_index_key='aug_index'):
    super().__init__()
    self._size = size
    self._video_key = video_key
    self._aug_index_key = aug_index_key

  def forward(self, images, boxes):
    images, boxes = uniform_crop_with_boxes(images[self._video_key], self._size, images[self._aug_index_key], boxes)


# TODO
class ApplyTransformsToVideoAndBox:
  def __init__(self, transforms):
    self._transforms = transforms

  def __call__(self, images, boxes):
    pass
    # for transform in self._transforms:
    #   if isinstance(transform, x) for x in [RandomCrop, RandomResizedCrop, CenterCrop, RandomHorizontalFlip, ShortSideScale]
