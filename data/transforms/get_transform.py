from .mvit import get_mvit_transforms
from .slowfast import get_slowfast_transforms
from .s3d import get_s3d_transforms


def get_transform(cfg):
  if cfg.net == 'mvit':
    return get_mvit_transforms(cfg)
  elif cfg.net == 'slowfast':
    return get_slowfast_transforms(cfg)
  elif cfg.net == 's3d':
    return get_s3d_transforms(cfg)
  else:
    raise NotImplementedError
