from .moma import MOMADataModule


def get_data(moma, cfg):
  return MOMADataModule(moma, cfg)
