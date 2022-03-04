from .backbone import get_mvit_backbone, get_slowfast_backbone
from .one_head_cls import OneHeadVideoClsModule
from .tri_head_cls import TriHeadVideoClsModule


def get_model(moma, cfg):
  if cfg.level == 'act':
    num_classes = cfg.num_classes_act
  elif cfg.level == 'sact':
    num_classes = cfg.num_classes_sact
  else:
    assert cfg.level == 'both'
    num_classes = (cfg.num_classes_act, cfg.num_classes_sact)

  if cfg.backbone == 'mvit':
    backbone = get_mvit_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')
  else:
    assert cfg.backbone == 'slowfast'
    backbone = get_slowfast_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')

  if cfg.level == 'act' or cfg.level == 'sact':
    return OneHeadVideoClsModule(moma, backbone, cfg)
  else:
    assert cfg.level == 'both'
    return TriHeadVideoClsModule(moma, backbone, cfg)
