import os
from dataset.us_city_dataset import USCityDataModule
from model.base import Net
from model.rnn import RNNSeqNetV2
from model.transformer import TransNetV2
from model.model_tft import TemporalFusionTransformer
from configs.configuration import CONFIGS
from configs.us_city_config import TRAIN_CONFIGS

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


def train(config):
    logger = TensorBoardLogger(
        '', name='us_city', version='rnn_v2_hidden128_dropout8e-1')

    save_best, save_last = setting_logger()
    # train the model
    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1, precision=config.precision,
                        accelerator='gpu', devices=config.devices, callbacks=[save_last, save_best])

    # init datamodule
    dm = USCityDataModule(scaling=config.scaling, input_ts=input_seq_len, cli_dir=config.cli_dir, res_dir=config.res_dir,
                          bud_dir=config.bud_dir, ref_csv=config.ref_csv, num_workers=config.num_workers,
                          batch_size=128)
    dm.setup()

    # init model
    model = RNNSeqNetV2(input_dim=input_dim, input_ts=input_seq_len,
                        output_ts=input_seq_len, hidden_dim=128, dropout=0.8, scaling=config.scaling)

    # train the model
    trainer.fit(model, datamodule=dm,
                ckpt_path='us_city/rnn_v2_hidden128_dropout8e-1/checkpoints/last.ckpt')

    # test the model
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    host_name = os.popen('hostname').read().strip()
    if 'tacc' in host_name:
        config = TRAIN_CONFIGS['tacc']
    else:
        config = TRAIN_CONFIGS['local']
    # normal train
    train(config)
