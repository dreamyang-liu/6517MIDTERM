import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from model import SegmentationModule
from data import NeuroDataModule
from utils import draw_mask_for_list
xd = NeuroDataModule(64)
model = SegmentationModule(1)
ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
train_trainer = pl.Trainer(accelerator='gpu', 
                            devices=3, 
                            max_epochs=100, 
                            strategy=ddp,
                            check_val_every_n_epoch=3,
                            reload_dataloaders_every_n_epochs=2,
                            log_every_n_steps=10,
                            auto_lr_find=True,
                            )
train_trainer.fit(model=model, datamodule=xd)
train_trainer.test(model=model, datamodule=xd)
dataloader = xd.test_dataloader()
for batch in dataloader:
    x, y = batch
    # breakpoint()
    y_hat = model.predict(x)
    draw_mask_for_list(x, y, y_hat)
    break