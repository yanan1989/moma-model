from .mvit import get_mvit_backbone
from .slowfast import get_slowfast_backbone
from .s3d import get_s3d_backbone


def get_backbone(num_classes, cfg):
  if cfg.backbone == 'mvit':
    backbone = get_mvit_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')
  elif cfg.backbone == 'slowfast':
    backbone = get_slowfast_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')
  else:
    assert cfg.backbone == 's3d'
    backbone = get_s3d_backbone(num_classes, cfg.dir_weights, cfg.strategy == 'finetune')
  return backbone
