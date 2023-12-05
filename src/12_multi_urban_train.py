import argparse
from dataset.dataset import MultiUrbanCitySim
from model.base import Net
from model.rnn import RNNSeqNetV2
from model.transformer import TransNetV2
from model.model_tft import TemporalFusionTransformer
from configuration import CONFIGS

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('high')
L.seed_everything(1340)

input_dim = 26
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

def train():
    logger = TensorBoardLogger('', name='multi_urban', version='rnn_v2_hidden32_dropout8e-1')

    save_best, save_last = setting_logger()
    # train the model
    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                        callbacks=[save_last, save_best])

    # init datamodule
    dm = MultiUrbanCitySim(input_ts=input_seq_len, output_ts=input_seq_len)
    dm.setup(stage='fit')

    # init model
    # model = RNNSeqNet(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len)
    # model = RNNEmbedNet(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len)
    # model = TransNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len)
    model = RNNSeqNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len, hidden_dim=32, dropout=0.8)

    # train the model
    trainer.fit(model, datamodule=dm)
    dm.setup(stage='test')

    # test the model
    trainer.test(model, datamodule=dm)

if __name__ == '__main__':
    # normal train
    train()