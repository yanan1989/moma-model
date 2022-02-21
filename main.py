import hydra
from omegaconf import DictConfig, OmegaConf

from data_modules import MOMADataModule


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))

  data_module = MOMADataModule(cfg.data_module)
  data_module.setup()

  for data in data_module.dataset_train:
    print(data['video'])
    break



if __name__ == '__main__':
  main()
