from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .utils import make_datasets


class MOMATwoLevelDataModule(LightningDataModule):
  def __init__(self, moma, cfg) -> None:
    super().__init__()
    self.moma = moma
    self.cfg = cfg

  def setup(self, stage=None):
    self.dataset_act_train, self.dataset_act_val, is_ddp_act = make_datasets(self.moma, 'act', self.cfg)
    self.dataset_sact_train, self.dataset_sact_val, is_ddp_sact = make_datasets(self.moma, 'sact', self.cfg)
    print(f'training set size: [act={self.dataset_act_train.num_videos}, '
          f'sact={self.dataset_sact_train.num_videos}], '
          f'validation set size: [act={self.dataset_act_val.num_videos}, '
          f'sact={self.dataset_sact_val.num_videos}]')
    print(f'is_ddp: {is_ddp_act and is_ddp_sact}')

  def train_dataloader(self):
    dataloader_act = DataLoader(self.dataset_act_train,
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=True)
    dataloader_sact = DataLoader(self.dataset_sact_train,
                                 batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    return {'act': dataloader_act, 'sact': dataloader_sact}

  def val_dataloader(self):
    dataloader_act = DataLoader(self.dataset_act_val,
                                batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                num_workers=self.cfg.num_workers,
                                pin_memory=True,
                                drop_last=False)
    dataloader_sact = DataLoader(self.dataset_sact_val,
                                 batch_size=int(self.cfg.batch_size/len(self.cfg.gpus)),
                                 num_workers=self.cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False)

    return [dataloader_act, dataloader_sact]
