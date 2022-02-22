import hydra
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

from data import MOMADataModule
from models import VideoClassificationModule
from trainers import get_trainer

level = 'sact'
backbone = 'slowfast'
strategy = 'finetune'


@hydra.main(config_path='configs', config_name=f'config_{level}_{backbone}_{strategy}')
def main(cfg: DictConfig) -> None:
  # print(OmegaConf.to_yaml(cfg))
  pprint(OmegaConf.to_container(cfg, resolve=True))

  data = MOMADataModule(cfg.data)
  model = VideoClassificationModule(cfg.model)
  trainer = get_trainer(cfg.trainer)

  trainer.fit(model, data)


if __name__ == '__main__':
  main()
