from .mvit import get_mvit
from .slowfast import get_slowfast
from .s3d import get_s3d


def get_net(cfg):
  num_classes = [cfg.num_classes[level] for level in cfg.levels]

  if cfg.net == 'mvit':
    net = get_mvit(num_classes, cfg.dir_weights, cfg.mode == 'finetune')
  elif cfg.net == 'slowfast':
    net = get_slowfast(num_classes, cfg.dir_weights, cfg.mode == 'finetune')
  elif cfg.net == 's3d':
    net = get_s3d(num_classes, cfg.dir_weights, cfg.mode == 'finetune')
  else:
    raise NotImplementedError

  return net
