from dataset.dataset import CitySimDataModule
from model.model import Net, RNNNet

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('highest')

training_version = 'ts_v0'
input_dim = 25
input_seq_len = 24

# set up the logger
logger = TensorBoardLogger('', version=training_version)

# store the best model and the last model
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

# train the model
trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                    callbacks=[save_last, save_best])

# init datamodule
dm = CitySimDataModule(input_ts=input_seq_len)
dm.setup(stage='fit')

# init model
# model = Net()
model = RNNNet(input_dim=input_dim, input_ts=input_seq_len)

# train the model
trainer.fit(model, datamodule=dm)
dm.setup(stage='test')

# test the model
trainer.test(model, datamodule=dm)
# trainer.predict(model, datamodule=dm)
