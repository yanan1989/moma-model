import os.path as osp
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy


def get_trainer(cfg):
  name = f"{'_'.join(cfg.levels)}_{cfg.net}_{cfg.mode}"+('_fs' if cfg.few_shot else '')
  logger = WandbLogger(
    project='moma',
    name=name,
    log_model='all',
    save_dir=cfg.dir_wandb
  )
  logger.log_hyperparams(cfg)

  kwargs = {
    'max_epochs': cfg.num_epochs,
    'logger': logger,
    'callbacks': [
      LearningRateMonitor(logging_interval='step'),
      ModelCheckpoint(every_n_epochs=5, save_last=True, dirpath=osp.join(cfg.dir_weights, f'ckpt/{name}'))
    ],
    'check_val_every_n_epoch': 1,
    'num_sanity_val_steps': 2,
    'log_every_n_steps': 10,
    'precision': 16,
    'accelerator': 'gpu',
    'devices': cfg.gpus,
    'strategy': DDPStrategy(find_unused_parameters=False) if len(cfg.gpus) > 1 else None,
    'detect_anomaly': True
  }

  trainer = Trainer(**kwargs)
  return trainer
