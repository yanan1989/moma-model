from .nets import get_net
from .one_head_classifier import OneHeadClassifierModule
from .dual_head_classifier import DualHeadClassifierModule


def get_model(moma, cfg):
  net = get_net(cfg)

  if len(cfg.levels) == 1:
    return OneHeadClassifierModule(moma, net, cfg)
  elif len(cfg.levels) == 2:
    return DualHeadClassifierModule(moma, net, cfg)
  else:
    raise NotImplementedError
