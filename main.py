import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:
  print(OmegaConf.to_yaml(cfg))
  # print(OmegaConf.to_container(cfg, resolve=True))

  data = instantiate(cfg.data, _recursive_=False)
  model = instantiate(cfg.model, _recursive_=False)
  trainer = instantiate(cfg.trainer)

  trainer.fit(model, data)


if __name__ == '__main__':
  main()
