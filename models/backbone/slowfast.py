import os
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.slowfast import create_slowfast
import torch
import torch.nn as nn

from .multi_head import MultiHead


def create_multi_head(in_features, out_features, pool, output_size, dropout_rate, activation, output_with_global_average):
  heads = []
  for out_feature in out_features:
    heads.append(create_res_basic_head(in_features=in_features,
                                       out_features=out_feature,
                                       pool=pool,
                                       output_size=output_size,
                                       dropout_rate=dropout_rate,
                                       activation=activation,
                                       output_with_global_average=output_with_global_average)
                 )
  multi_head = MultiHead(nn.ModuleList(heads))
  return multi_head


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/module/model/slowfast_r50.yaml
def get_slowfast_backbone(num_classes, dir_weights, finetune):
  module = create_slowfast(
    input_channels=(3, 3),
    model_depth=50,
    model_num_class=num_classes[0] if len(num_classes) == 1 else num_classes,
    dropout_rate=0.5,
    slowfast_fusion_conv_kernel_size=(7, 1, 1),
    head=create_res_basic_head if len(num_classes) == 1 else create_multi_head
  )

  if finetune:
    weights = torch.load(os.path.join(dir_weights, 'pretrain/SLOWFAST_8x8_R50.pyth'))['model_state']
    weights.pop('blocks.6.proj.weight', None)
    weights.pop('blocks.6.proj.bias', None)
    print(f'{list(set(module.state_dict())-set(weights.keys()))} will be trained from scratch')
    module.load_state_dict(weights, strict=False)

  return module
