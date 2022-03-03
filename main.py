from momaapi import MOMA
from omegaconf import OmegaConf

from data import get_data
from models import get_model
from trainers import get_trainer


def main() -> None:
  cfg = OmegaConf.load('./configs/config.yaml')
  # print(OmegaConf.to_yaml(cfg))

  moma = MOMA(cfg.dir_moma, small=True, toy=True)
  data = get_data(moma, cfg)
  model = get_model(moma, cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, data)


if __name__ == '__main__':
  main()
