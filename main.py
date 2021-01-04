from models.detr import build
from omegaconf.dictconfig import DictConfig
import pytorch_lightning as pl
from models.model import LitModel
import hydra
from hydra.utils import instantiate, call

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    datamodule = instantiate(cfg.datamodule)
    checkpoint = instantiate(cfg.checkpoint)
    model, criterion, postprocessors = build(cfg)
    model = LitModel(model, criterion, cfg.hparams)

    trainer = pl.Trainer.from_argparse_args(cfg.trainer, callbacks=[checkpoint])
    trainer.fit(model, datamodule=datamodule)
    
    if cfg.experiment.test:
        trainer.test(model, datamodule)

if __name__ == "__main__":
    main()
