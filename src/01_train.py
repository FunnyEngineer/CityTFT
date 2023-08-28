from dataset import CitySimDataModule
from model import Net
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('highest')


# Define a lightning module: neural network

# Define a dataset

logger = TensorBoardLogger('', version='normalized_load_2')
save_best = ModelCheckpoint(
    save_top_k=5,
    monitor='val/total_loss',
    mode='min',
    save_last=True,
    filename='epoch={epoch:02d}-step={global_step}-val_loss={val/total_loss:.2f}',
    auto_insert_metric_name=False,
)
save_last = ModelCheckpoint(
    save_top_k=5,
    monitor="global_step",
    mode="max",
    filename='epoch={epoch:02d}-step={global_step}-val_loss={val/total_loss:.2f}',
    auto_insert_metric_name=False,
)
trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                    callbacks=[save_last, save_best])
dm = CitySimDataModule()
dm.setup(stage='fit')
model = Net()
trainer.fit(model, datamodule=dm)
dm.setup(stage='test')
trainer.test(model, datamodule=dm)
# trainer.predict(model, datamodule=dm)
