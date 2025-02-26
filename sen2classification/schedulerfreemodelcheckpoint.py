from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from schedulefree import AdamWScheduleFree


class SchedulerFreeModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        opts = trainer.optimizers
        if not isinstance(opts, list):
            opts = [opts]

        # save current train/eval state so that we can restore them
        states = []
        for opt in opts:
            if isinstance(opt, AdamWScheduleFree):
                # probe optimizer state by checking the train state of the first parameter group
                is_train_mode = opt.param_groups[0]["train_mode"]
                if is_train_mode:
                    opt.eval()
            else:
                is_train_mode = None
            states.append(is_train_mode)

        super()._save_checkpoint(trainer, filepath)

        # restore previous state
        for (opt, wasintrainmode) in zip(opts, states):
            if isinstance(opt, AdamWScheduleFree):
                if wasintrainmode:
                    opt.train()
