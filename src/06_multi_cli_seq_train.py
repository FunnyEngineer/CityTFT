from darts.models.forecasting.tft_model import TFTModel
import argparse
from dataset.dataset import CitySimDataModule
from dataset.tft_dataset import CitySimTFTDataModule
from model.model import Net, RNNSeqNet, RNNEmbedNet, TransformerSeqNet, HybridRNNAttenNet, TransNetV2
from model.model_tft import TemporalFusionTransformer
from configuration import CONFIGS

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
        save_top_k=3,
        monitor='val/total_loss',
        mode='min',
        save_last=True,
        filename='epoch={epoch:02d}-step={global_step}-val_loss={val/total_loss:.8f}',
        auto_insert_metric_name=False,
    )
    save_last = ModelCheckpoint(
        save_top_k=3,
        monitor="global_step",
        mode="max",
        filename='epoch={epoch:02d}-step={global_step}-val_loss={val/total_loss:.8f}',
        auto_insert_metric_name=False,
    )
    return save_best, save_last


def RNN_train():
    logger = TensorBoardLogger('multi_cli_log', version='rnn_seq_v1')

    save_best, save_last = setting_logger()
    # train the model
    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                        callbacks=[save_last, save_best])

    # init datamodule
    dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len)
    dm.setup(stage='fit')

    # init model
    model = RNNSeqNet(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len)
    # model = RNNEmbedNet(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len)

    # train the model
    trainer.fit(model, datamodule=dm)
    dm.setup(stage='test')

    # test the model
    trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    RNN_train()