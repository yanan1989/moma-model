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


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_slowfast.yaml
def get_slowfast_transforms(T=8, alpha=4):
  transform_train = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T*alpha),
        Div255(),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop(224),
        RandomHorizontalFlip(p=0.5),
        SlowFastPackPathway(alpha=alpha)
      ]
    )
  )

  transform_val = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T*alpha),
        Div255(),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ShortSideScale(size=256),
        CenterCrop(256),
        SlowFastPackPathway(alpha=alpha)
      ]
    )
  )
