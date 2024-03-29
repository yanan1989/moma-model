from momaapi import MOMA
from omegaconf import OmegaConf

from data import get_data
from models import get_model
from trainers import get_trainer


def main():
  cfg = OmegaConf.load('configs/mvit.yaml')

  moma = MOMA(cfg.dir_moma)
  data = get_data(moma, cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, datamodule=data)


if __name__ == '__main__':
  main()
