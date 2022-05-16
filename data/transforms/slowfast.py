from pytorchvideo.transforms import (
  ApplyTransformToKey,
  Div255,
  Normalize,
  RandomShortSideScale,
  ShortSideScale,
  UniformTemporalSubsample
)
from pytorchvideo_trainer.datamodule.transforms import SlowFastPackPathway
from torchvision.transforms import (
  CenterCrop,
  Compose,
  RandomCrop,
  RandomHorizontalFlip
)

from .constants import *


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_slowfast.yaml
def get_slowfast_transforms(cfg):
  transform_train = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=cfg.T*cfg.alpha),
        Div255(),
        Normalize(mean=MEAN_KINETICS, std=STD_KINETICS),
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop(224),
        RandomHorizontalFlip(p=0.5),
        SlowFastPackPathway(alpha=cfg.alpha)
      ]
    )
  )

  transform_val = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=cfg.T*cfg.alpha),
        Div255(),
        Normalize(mean=MEAN_KINETICS, std=STD_KINETICS),
        ShortSideScale(size=256),
        CenterCrop(256),
        SlowFastPackPathway(alpha=cfg.alpha)
      ]
    )
  )

  transform_test = transform_val

  return transform_train, transform_val, transform_test
