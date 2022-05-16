from omegaconf.dictconfig import DictConfig

from .one_head_classifier import OneHeadClassifierModule
from .dual_head_classifier import DualHeadClassifierModule


def get_model(cfg):
  if isinstance(cfg.lr, DictConfig):
    cfg.lr = cfg.lr[cfg.optimizer]

  if isinstance(cfg.wd, DictConfig):
    cfg.wd = cfg.wd[cfg.optimizer]

  if len(cfg.levels) == 1:
    return OneHeadClassifierModule(cfg)
  elif len(cfg.levels) == 2:
    return DualHeadClassifierModule(cfg)
  else:
    raise NotImplementedError
