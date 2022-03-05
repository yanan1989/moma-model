from .backbone import get_backbone
from .one_head_cls import OneHeadVideoClsModule
from .dual_head_cls import DualHeadVideoClsModule


def get_model(moma, cfg):
  num_classes = [cfg.num_classes[level] for level in cfg.levels]

  backbone = get_backbone(num_classes, cfg)

  if len(num_classes) == 1:
    return OneHeadVideoClsModule(moma, backbone, cfg)
  else:
    assert len(num_classes) == 2
    return DualHeadVideoClsModule(moma, backbone, cfg)
