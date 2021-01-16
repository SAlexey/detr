from models.detr import build
from omegaconf.dictconfig import DictConfig
from models.model import LitModel
from pathlib import Path
import hydra
from hydra.utils import instantiate, call, to_absolute_path

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    datamodule = instantiate(cfg.datamodule)
    checkpoint = instantiate(cfg.checkpoint)

    resume = None
    
    if cfg.experiment.resume:
        checkpoints = sorted(Path(to_absolute_path("outputs")).rglob(f"checkpoints/**/{cfg.experiment.name}/**/*.ckpt"))
        if checkpoints:
            resume = str(checkpoints[-1])

    model, criterion, _ = build(cfg)
    model = LitModel(model, criterion, cfg.hparams)

    trainer = instantiate(cfg.trainer, callbacks=[checkpoint], resume_from_checkpoint=resume)
    trainer.fit(model, datamodule=datamodule)
    
    if cfg.experiment.test:
        trainer.test(model, datamodule)

if __name__ == "__main__":
    main()
