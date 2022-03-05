import os
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers
import torch
import torch.nn as nn

from .multi_head import MultiHead


def create_multi_head(in_features, out_features, seq_pool_type, dropout_rate, activation):
  heads = []
  for out_feature in out_features:
    heads.append(create_vit_basic_head(in_features=in_features,
                                       out_features=out_feature,
                                       seq_pool_type=seq_pool_type,
                                       dropout_rate=dropout_rate,
                                       activation=activation)
                 )
  multi_head = MultiHead(nn.ModuleList(heads))
  return multi_head


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/module/model/mvit_base_16x4.yaml
def get_mvit_backbone(num_classes, dir_weights, finetune):
  module = create_multiscale_vision_transformers(
    spatial_size=224,
    temporal_size=16,
    cls_embed_on=True,
    sep_pos_embed=True,
    depth=16,
    norm='layernorm',
    input_channels=3,
    patch_embed_dim=96,
    conv_patch_embed_kernel=(3, 7, 7),
    conv_patch_embed_stride=(2, 4, 4),
    conv_patch_embed_padding=(1, 3, 3),
    enable_patch_embed_norm=False,
    use_2d_patch=False,
    # Attention block config,
    num_heads=1,
    mlp_ratio=4.0,
    qkv_bias=True,
    dropout_rate_block=0.0,
    droppath_rate_block=0.2,
    pooling_mode='conv',
    pool_first=False,
    embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    pool_kv_stride_size=None,
    pool_kv_stride_adaptive=[1, 8, 8],
    pool_kvq_kernel=[3, 3, 3],
    # Head config,
    head=create_vit_basic_head if len(num_classes) == 1 else create_multi_head,
    head_dropout_rate=0.5,
    head_activation=None,
    head_num_classes=num_classes[0] if len(num_classes) == 1 else num_classes
  )

  if finetune:
    weights = torch.load(os.path.join(dir_weights, 'pretrain/MVIT_B_16x4.pyth'))['model_state']
    weights.pop('head.proj.weight', None)
    weights.pop('head.proj.bias', None)
    print(f'{list(set(module.state_dict().keys())-set(weights.keys()))} will be trained from scratch')
    module.load_state_dict(weights, strict=False)

  return module
