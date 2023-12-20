import argparse
from dataset.dataset import CitySimDataModule
from model.base import Net
from model.rnn import RNNSeqNetV2
from model.transformer import TransNetV2
from model.model_tft import TemporalFusionTransformer
from configs.configuration import CONFIGS

from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
torch.set_float32_matmul_precision('high')
L.seed_everything(1340)

training_version = 'rnn_seq_v0'
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
    logger = TensorBoardLogger('', name='multi_cli', version='rnn_v2_hidden32_dropout8e-1')

    save_best, save_last = setting_logger()
    # train the model
    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                        callbacks=[save_last, save_best])

    # init datamodule
    dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len)
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

def validate():
    trainer = L.Trainer()

    dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len)
    dm.setup(stage='fit')
    model = RNNSeqNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len).load_from_checkpoint('multi_cli/rnn_with_prob_quantile_v3_adamW_lr1e-4/checkpoints/epoch=227-step=2075447.125-val_loss=2.16468382.ckpt')

    trainer.validate(model, datamodule=dm)

def TFT_train(args):
    logger = TensorBoardLogger('', name='multi_cli', version='tft_with_prob_v3_without_target_adamW_lr1e-4')
    save_best, save_last = setting_logger()

    trainer = L.Trainer(max_epochs=400, logger=logger, check_val_every_n_epoch=1,
                        callbacks=[save_last, save_best])

    dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len, mode='tft')

    dm.setup(stage='fit')
    # darts version
    # pytorch version
    config = CONFIGS[args.dataset]()
    model = TemporalFusionTransformer(config)

    trainer.fit(model, datamodule=dm)
    dm.setup(stage='test')

    # test the model
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    # normal train
    train()

    # validate
    # validate()

    # tft train
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--encoder_length', type=int, default=24)
    # parser.add_argument('--dataset', type=str, required=True, choices=CONFIGS.keys(),
    #                     help='Dataset name', default='citysim')
    # parser.add_argument('--epochs', type=int, default=25,
    #                     help='Default number of training epochs')
    # parser.add_argument('--sample_data', type=lambda x: int(float(x)), nargs=2, default=[-1, -1],
    #                     help="""Subsample the dataset. Specify number of training and valid examples.
    #                     Values can be provided in scientific notation. Floats will be truncated.""")
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision')
    # parser.add_argument('--clip_grad', type=float, default=0.0)
    # parser.add_argument('--grad_accumulation', type=int, default=0)
    # parser.add_argument('--early_stopping', type=int, default=1000,
    #                     help='Stop training if validation loss does not improve for more than this number of epochs.')
    # parser.add_argument('--results', type=str, default='/results',
    #                     help='Directory in which results are stored')
    # parser.add_argument('--log_file', type=str, default='dllogger.json',
    #                     help='Name of dllogger output file')
    # parser.add_argument('--overwrite_config', type=str, default='',
    #                     help='JSON string used to overload config')
    # parser.add_argument('--affinity', type=str,
    #                     default='socket_unique_interleaved',
    #                     choices=['socket', 'single', 'single_unique',
    #                              'socket_unique_interleaved',
    #                              'socket_unique_continuous',
    #                              'disabled'],
    #                     help='type of CPU affinity')
    # parser.add_argument("--ema_decay", type=float, default=0.0,
    #                     help='Use exponential moving average')
    # parser.add_argument("--disable_benchmark", action='store_true',
    #                     help='Disable benchmarking mode')
    # ARGS = parser.parse_args()
    # TFT_train(ARGS)