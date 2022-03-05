from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .utils import make_datasets


class MOMADataModule(LightningDataModule):
  def __init__(self, moma, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg
    self.datasets_train, self.datasets_val, self.datasets_test = {}, {}, {}

  def setup(self, stage=None):
    for level in self.cfg.levels:
      self.datasets_train[level], self.datasets_val[level], self.datasets_test[level] = \
          make_datasets(self.moma, level, self.cfg)
      print(f'Level {level}: '
            f'train={self.datasets_train[level].num_videos}, '
            f'val={self.datasets_val[level].num_videos}, '
            f'test={self.datasets_test[level].num_videos}')

  def train_dataloader(self):
    loaders = {}
    for level in self.cfg.levels:
      loaders[level] = DataLoader(self.datasets_train[level],
                                  batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                  num_workers=self.cfg.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    return loaders[self.cfg.levels[0]] if len(loaders) == 1 else loaders

  def val_dataloader(self):
    loaders = []
    for level in self.cfg.levels:
      loaders.append(DataLoader(self.datasets_val[level],
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=False))
    return loaders[0] if len(loaders) == 1 else loaders

  def test_dataloader(self):
    loaders = []
    for level in self.cfg.levels:
      loaders.append(DataLoader(self.datasets_test[level],
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=False))
    return loaders[0] if len(loaders) == 1 else loaders
