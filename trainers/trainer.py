from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import warnings
warnings.filterwarnings('ignore', '.*upsampling*', )


def get_trainer(cfg):
  # Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L423
  trainer = Trainer(
    max_epochs=cfg.num_epochs,
    logger=WandbLogger(
      project='moma',
      name=f'{cfg.level}_{cfg.backbone}_{cfg.strategy}',
      log_model='all',
      save_dir=cfg.dir_wandb
    ),
    callbacks=[
      LearningRateMonitor(logging_interval='step'),
      ModelCheckpoint(monitor='val/acc', mode='max', dirpath=cfg.dir_weights)
    ],
    precision=16,
    num_sanity_val_steps=0,
    log_every_n_steps=10,
    gpus=cfg.gpus,
    **(
      {
        'strategy': 'ddp',
        'replace_sampler_ddp': False
       } if len(cfg.gpus) > 1 else {}
    )
  )

  return trainer
