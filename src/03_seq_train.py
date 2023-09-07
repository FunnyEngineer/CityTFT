from darts.models.forecasting.tft_model import TFTModel

from dataset import CitySimDataModule
from model import Net, RNNSeqNet

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('highest')


training_version = 'rnn_seq_v0'
input_dim = 25
input_seq_len = 24

def setting_logger():
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
    return save_best, save_last

def RNN_train():
    logger = TensorBoardLogger('', version=training_version)

    save_best, save_last = setting_logger()
    # train the model
    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                    callbacks=[save_last, save_best])
    
    # init datamodule
    dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len)
    dm.setup(stage='fit')

    # init model
    model = RNNSeqNet(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len)

    # train the model
    trainer.fit(model, datamodule=dm)
    dm.setup(stage='test')

    # test the model
    trainer.test(model, datamodule=dm)

def TFT_train():
    save_best, save_last = setting_logger()
    dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len)
    dm.setup(stage='fit')
    TFTModel(input_chunk_length=input_dim, output_chunk_length=2, log_tensorboard=True,
         pl_trainer_kwargs={"callbacks": [save_best, save_last]})

def main():
    RNN_train()
    # TFT_train()

if __name__ == '__main__':
    main()