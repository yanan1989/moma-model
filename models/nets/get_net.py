def get_net(cfg):
  if cfg.net == 'mvit':
    from .mvit import get_mvit
    net = get_mvit(cfg)
  elif cfg.net == 'slowfast':
    from .slowfast import get_slowfast
    net = get_slowfast(cfg)
  elif cfg.net == 's3d':
    from .s3d import get_s3d
    net = get_s3d(cfg)
  else:
    raise NotImplementedError

  return net
