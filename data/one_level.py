from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .utils import make_datasets


class MOMAOneLevelDataModule(LightningDataModule):
  def __init__(self, moma, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg

  def setup(self, stage=None):
    self.dataset_train, self.dataset_val, is_ddp = make_datasets(self.moma, self.cfg.level, self.cfg)
    print(f'training set size: {self.dataset_train.num_videos}, '
          f'validation set size: {self.dataset_val.num_videos}')
    print(f'is_ddp: {is_ddp}')

  def train_dataloader(self):
    dataloader = DataLoader(self.dataset_train,
                            batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                            num_workers=self.cfg.num_workers,
                            pin_memory=True,
                            drop_last=True)
    return dataloader

  def val_dataloader(self):
    dataloader = DataLoader(self.dataset_val,
                            batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                            num_workers=self.cfg.num_workers,
                            pin_memory=True,
                            drop_last=False)
    return dataloader
