from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


def get_trainer(cfg: DictConfig):
  trainer = Trainer(
    max_epochs=cfg.max_epochs,
    logger=WandbLogger(
      project='moma',
      name=cfg.name,
      log_model='all',
      save_dir=cfg.dir_wandb
    ),
    callbacks=[
      LearningRateMonitor(logging_interval='epoch'),
      ModelCheckpoint(monitor='val/acc', mode='max', dirpath=cfg.dir_ckpt)
    ],
    gpus=cfg.gpus,
    **(
      {'strategy': DDPPlugin(find_unused_parameters=False), 'replace_sampler_ddp': False}
      if len(cfg.gpus) > 1 else {}
    )
  )

  return trainer
