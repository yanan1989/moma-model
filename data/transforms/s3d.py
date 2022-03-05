from pytorchvideo.transforms import (
  ApplyTransformToKey,
  Div255,
  Normalize,
  Permute,
  RandomResizedCrop,
  ShortSideScale,
  UniformTemporalSubsample
)

from torchvision.transforms import (
  CenterCrop,
  Compose,
  RandomHorizontalFlip
)


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_mvit_16x4.yaml
def get_s3d_transforms(T=30):
  transform_train = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T),
        Div255(),
        RandomResizedCrop(target_height=224, target_width=224, scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
        RandomHorizontalFlip(p=0.5)
      ]
    )
  )

  transform_val = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T),
        Div255(),
        ShortSideScale(224),
        CenterCrop(224)
      ]
    )
  )

  transform_test = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T),
        Div255(),
        ShortSideScale(224),
        CenterCrop(224)
      ]
    )
  )

  return transform_train, transform_val, transform_test
