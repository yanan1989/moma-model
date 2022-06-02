from .moma import MOMADataModule


def get_data(cfg):
  data = MOMADataModule(cfg)
  return data
