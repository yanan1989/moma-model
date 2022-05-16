from .moma import MOMADataModule


def get_data(moma, cfg):
  data = MOMADataModule(moma, cfg)
  cfg.num_classes = [data.num_classes[level] for level in cfg.levels]  # TODO: support few-shot
  return data
