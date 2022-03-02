import os
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.slowfast import create_slowfast
import torch
import torch.nn as nn

from .tri_head import TriHead


def create_tri_head(in_features, out_features, pool, output_size, dropout_rate, activation, output_with_global_average):
  head1 = create_res_basic_head(in_features=in_features,
                                out_features=out_features[0],
                                pool=pool,
                                output_size=output_size,
                                dropout_rate=dropout_rate,
                                activation=activation,
                                output_with_global_average=output_with_global_average)
  head2 = create_res_basic_head(in_features=in_features,
                                out_features=1,
                                pool=pool,
                                output_size=output_size,
                                dropout_rate=dropout_rate,
                                activation=activation,
                                output_with_global_average=output_with_global_average)
  head3 = create_res_basic_head(in_features=in_features,
                                out_features=out_features[1],
                                pool=pool,
                                output_size=output_size,
                                dropout_rate=dropout_rate,
                                activation=activation,
                                output_with_global_average=output_with_global_average)
  tri_head = TriHead(nn.ModuleList([head1, head2, head3]))
  return tri_head


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/module/model/slowfast_r50.yaml
def get_slowfast_backbone(num_classes, dir_weights, finetune):
  module = create_slowfast(
    input_channels=(3, 3),
    model_depth=50,
    model_num_class=num_classes,
    dropout_rate=0.5,
    slowfast_fusion_conv_kernel_size=(7, 1, 1),
    head=create_tri_head if isinstance(num_classes, tuple) else create_res_basic_head
  )

  if finetune:
    weights = torch.load(os.path.join(dir_weights, 'pretrain/SLOWFAST_8x8_R50.pyth'))['model_state']
    weights.pop('blocks.6.proj.weight', None)
    weights.pop('blocks.6.proj.bias', None)
    print(f'{list(set(module.state_dict())-set(weights.keys()))} will be trained from scratch')
    module.load_state_dict(weights, strict=False)
  return module
