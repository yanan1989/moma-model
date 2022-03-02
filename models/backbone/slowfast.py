import os
from pytorchvideo.models.slowfast import create_slowfast
import torch


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/module/model/slowfast_r50.yaml
def get_slowfast_backbone(num_classes, dir_weights, finetune):
  module = create_slowfast(
    input_channels=(3, 3),
    model_depth=50,
    model_num_class=num_classes,
    dropout_rate=0.5,
    slowfast_fusion_conv_kernel_size=(7, 1, 1)
  )

  if finetune:
    weights = torch.load(os.path.join(dir_weights, 'pretrain/SLOWFAST_8x8_R50.pyth'))['model_state']
    weights.pop('blocks.6.proj.weight', None)
    weights.pop('blocks.6.proj.bias', None)
    weights.pop('head.proj.weight', None)
    weights.pop('head.proj.bias', None)
    print(f'{list(set(module.state_dict())-set(weights.keys()))} will be trained from scratch')
    module.load_state_dict(weights, strict=False)
  return module
