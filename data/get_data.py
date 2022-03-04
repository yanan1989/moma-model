from .one_level import MOMAOneLevelDataModule
from .two_level import MOMATwoLevelDataModule


def get_data(moma, cfg):
  if cfg.level in ['act', 'sact']:
    return MOMAOneLevelDataModule(moma, cfg)
  else:
    assert cfg.level == 'both'
    return MOMATwoLevelDataModule(moma, cfg)
