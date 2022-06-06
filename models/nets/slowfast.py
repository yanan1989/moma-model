import os.path as osp
from pytorchvideo.models.head import create_res_basic_head
from pytorchvideo.models.slowfast import create_slowfast
import torch
import torch.nn as nn
from torchvision.datasets.utils import download_url

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
def get_slowfast(cfg):
  net = create_slowfast(
    model_num_class=cfg.num_classes[0] if len(cfg.num_classes) == 1 else cfg.num_classes,
    head=create_res_basic_head if len(cfg.num_classes) == 1 else create_multi_head
  )

  if cfg.mode == 'from_scratch':
    print('Initializing randomly')

  elif cfg.mode == 'finetune':
    if cfg.weight == 'ckpt':
      print('Loading checkpoint')
      weight = torch.load(osp.join(cfg.dir_weights, cfg.rpath_ckpt))['state_dict']
      weight = {k.removeprefix('net.'): v for k, v in weight.items()}
      weight.pop('blocks.6.proj.weight')
      weight.pop('blocks.6.proj.bias')
      keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
      assert len(keys_unexpected) == 0
      print(f'{keys_missing} will be trained from scratch')

    elif cfg.weight == 'pretrain':
      print('Loading Kinetics pre-trained weight')
      dir_pretrain = osp.join(cfg.dir_weights, 'pretrain')
      fname_pretrain = 'SLOWFAST_8x8_R50.pyth'
      if not osp.exists(osp.join(dir_pretrain, fname_pretrain)):
        download_url(f'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/{fname_pretrain}', dir_pretrain)

      weight = torch.load(osp.join(dir_pretrain, fname_pretrain))['model_state']
      weight.pop('blocks.6.proj.weight')
      weight.pop('blocks.6.proj.bias')
      keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
      assert len(keys_unexpected) == 0
      print(f'{keys_missing} will be trained from scratch')

    else:
      raise NotImplementedError

  else:
    raise NotImplementedError

  return net
