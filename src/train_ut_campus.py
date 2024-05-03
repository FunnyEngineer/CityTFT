import os
from dataset.dataset import CitySimDataModule
from model.base import Net
from model.rnn import LSTMSeq, LSTMSeqQuantile, LSTMSeqProb
from model.transformer import TransNetV2, TransSeq, TransSeqQuantile, TransSeqProb
from model.model_tft import TemporalFusionTransformer
from model.tft import UninterpretableTFT
from configs.configuration import CONFIGS
from configs.us_city_config import TRAIN_CONFIGS

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('high')
L.seed_everything(1340)

training_version = 'rnn_seq_v0'
input_dim = 26
seq_len = 24


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
    # model_dict = {
    #     'rnn_mse': LSTMSeq(input_dim=input_dim, input_seq_len=seq_len, output_ts=seq_len,
    #                        output_seq_len=seq_len, encode_layers=2, decode_layers=2,
    #                         hidden_dim=32, dropout=0.8, scaling=config.scaling),
    #     'rnn_prob': RNNSeqNetV2,
    #     'transformer_mse': TransNetV2,
    #     'transformer_prob': TransNetV2,
    #     'tft_mse': TemporalFusionTransformer,
    #     'tft_prob': TemporalFusionTransformer,
    # }

    logger = TensorBoardLogger('', name='ut_campus',
                               version='trans_prob_hidden64_dropout8e-1_loadDiffL1_focalLoss')

    save_best, save_last = setting_logger()

    # train the model
    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                        callbacks=[save_last, save_best])

    # init datamodule
    dm = CitySimDataModule(
        input_ts=seq_len, output_ts=seq_len, scaling=config.scaling)
    dm.setup(stage='fit')

    # init model
    # rnn_mse
    # model = LSTMSeq(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
    #                 encode_layers=2, decode_layers=2,
    #                 hidden_dim=64, dropout=0.8, scaling=config.scaling)

    # rnn_quantile
    # model = LSTMSeqQuantile(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
    #                 encode_layers=2, decode_layers=2,
    #                 hidden_dim=64, dropout=0.8, scaling=config.scaling)

    # rnn_prob
    # model = LSTMSeqProb(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
    #                 encode_layers=2, decode_layers=2,
    #                 hidden_dim=64, dropout=0.8, scaling=config.scaling)

    # transformer_mse
    # model = TransSeq(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
    #                 encode_layers=2, decode_layers=2,
    #                 hidden_dim=64, dropout=0.8, scaling=config.scaling)

    # transformer_quantile
    # model = TransSeqQuantile(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
    #                 encode_layers=2, decode_layers=2,
    #                 hidden_dim=64, dropout=0.8, scaling=config.scaling)

    # transformer_prob
    model = TransSeqProb(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
                         encode_layers=2, decode_layers=2,
                         hidden_dim=64, dropout=0.8, scaling=config.scaling)

    # train the model
    trainer.fit(model, datamodule=dm)
    dm.setup(stage='test')

    # test the model
    trainer.test(model, datamodule=dm)


def validate():
    trainer = L.Trainer()

    dm = CitySimDataModule(input_ts=seq_len, output_ts=seq_len)
    dm.setup(stage='fit')
    model = RNNSeqNetV2(input_dim=input_dim, input_ts=seq_len, output_ts=seq_len).load_from_checkpoint(
        'multi_cli/rnn_with_prob_quantile_v3_adamW_lr1e-4/checkpoints/epoch=227-step=2075447.125-val_loss=2.16468382.ckpt')

    trainer.validate(model, datamodule=dm)


def TFT_train(args):
    logger = TensorBoardLogger(
        '', name='ut_campus', version='untft_prob_mseAll_hidden128_lr1e-5')
    save_best, save_last = setting_logger()

    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                        callbacks=[save_last, save_best])

    dm = CitySimDataModule(input_ts=seq_len,
                           output_ts=seq_len, mode='tft', scaling=args.scaling)

    dm.setup(stage='fit')
    # darts version
    # pytorch version
    config = CONFIGS['citysim']()
    # model = TemporalFusionTransformer(config, args.scaling)
    model = UninterpretableTFT(
        seq_len=24, 
        static_dim=13, 
        temporal_dim=13, 
        hidden_dim=128, 
        scaling=args.scaling,)

    trainer.fit(model, datamodule=dm)
    dm.setup(stage='test')

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
    # TFT_train(config)
