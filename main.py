import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from data import MOMADataModule
from models import SlowFastModule


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
  data = MOMADataModule(cfg.data_module)
  model = SlowFastModule(cfg.module)
  logger = WandbLogger(project=cfg.project, save_dir=cfg.dir_log)

  trainer = Trainer(gpus=cfg.gpu, strategy=cfg.strategy, max_epochs=cfg.module.max_epochs, replace_sampler_ddp=False,
                    logger=logger, plugins=DDPPlugin(find_unused_parameters=False))
  trainer.fit(model, data)


if __name__ == '__main__':
  main()
