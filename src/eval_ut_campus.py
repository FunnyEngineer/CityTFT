import os
from dataset.dataset import CitySimDataModule
from model.rnn import LSTMSeq, LSTMSeqQuantile, LSTMSeqProb
from model.transformer import TransNetV2, TransSeq, TransSeqQuantile, TransSeqProb
from model.model_tft import TemporalFusionTransformer
from model.tft import UninterpretableTFT
from configs.us_city_config import TRAIN_CONFIGS

import lightning as L
import torch
import pdb
import matplotlib.pyplot as plt
torch.set_float32_matmul_precision('high')
L.seed_everything(1340)

training_version = 'rnn_seq_v0'
input_dim = 26
seq_len = 24


def eval(config):
    dm = CitySimDataModule(
        input_ts=seq_len, output_ts=seq_len, scaling=config.scaling)
    dm.setup(stage='test')

    # load model
    model = TransSeq(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
                     encode_layers=2, decode_layers=2,
                     hidden_dim=64, dropout=0.8, scaling=config.scaling).load_from_checkpoint(
        'ut_campus/trans_mse_hidden64_dropout8e-1/checkpoints/epoch=398-step=3632096.0-val_loss=0.12867562.ckpt',
        input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
        encode_layers=2, decode_layers=2,
        hidden_dim=64, dropout=0.8, scaling=config.scaling)

    model2 = TransSeqProb(input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
                          encode_layers=2, decode_layers=2,
                          hidden_dim=64, dropout=0.8, scaling=config.scaling).load_from_checkpoint(
        'ut_campus/trans_prob_hidden64_dropout8e-1/checkpoints/epoch=397-step=3622943.0-val_loss=1.75731242.ckpt',
        input_dim=input_dim, input_seq_len=seq_len, output_seq_len=seq_len,
        encode_layers=2, decode_layers=2,
        hidden_dim=64, dropout=0.8, scaling=config.scaling)

    for i, batch in enumerate(dm.test_dataloader()):
        x, y = model.split_reshape(batch)
        if (y[:, :, 0] == model2.h_zero).sum() < 1000:
            x = x.to(model.device)
            heat_hat, cool_hat = model(x)
            heat_hat2, cool_hat2, heat_prob, cool_prob = model2(x)
            heat_pred2, cool_pred2 = model2.generate_load(
                heat_hat2, cool_hat2, heat_prob, cool_prob)
            heat_pred3, cool_pred3 = generate_load_sigmoid(
                model2, heat_hat2, cool_hat2, heat_prob, cool_prob)
            # inverse scaling
            heat_pred, cool_pred = model.inverse_transform_load(heat_hat, cool_hat)
            heat_true, cool_true = model.inverse_transform_load(
                y[:, :, 0], y[:, :, 1])
            for data_i in range(63, 53, -1):
                # init figure with 2 subplots
                fig, axs = plt.subplots(2)
                # plot two time series
                axs[0].plot(heat_true[data_i, :].cpu().numpy(),
                            label='Heat True',
                            color='red')
                axs[0].plot(heat_pred[data_i, :, 0].detach(
                ).cpu().numpy(), label='Trans MSE Pred', linestyle='dashdot')
                axs[0].plot(heat_pred2[data_i, :, 0].detach(
                ).cpu().numpy(), label='Trans Prob Pred', linestyle='dashdot')
                axs[0].plot(heat_pred3[data_i, :, 0].detach(
                ).cpu().numpy(), label='Trans ProbSig Pred', linestyle='dashdot')
                axs[1].plot(cool_true[data_i, :].cpu().numpy(),
                            label='Cool True',
                            color='blue')
                axs[1].plot(cool_pred[data_i, :, 0].detach(
                ).cpu().numpy(), label='Trans MSE Pred', linestyle='dashdot')
                axs[1].plot(cool_pred2[data_i, :, 0].detach(
                ).cpu().numpy(), label='Trans Prob Pred', linestyle='dashdot')
                axs[1].plot(cool_pred3[data_i, :, 0].detach(
                ).cpu().numpy(), label='Trans ProbSig Pred', linestyle='dashdot')
                axs[0].legend()
                axs[1].legend()
                plt.show()
            pdb.set_trace()


def generate_load_sigmoid(model, heat_hat, cool_hat, heat_prob, cool_prob):
    heat_hat, cool_hat = model.inverse_transform_load(heat_hat, cool_hat)
    heat_hat[(torch.sigmoid(heat_prob) < model.h_trigger_ratio).squeeze()] = 0
    cool_hat[(torch.sigmoid(cool_prob) < model.c_trigger_ratio).squeeze()] = 0
    return heat_hat, cool_hat


if __name__ == '__main__':
    host_name = os.popen('hostname').read().strip()
    if 'tacc' in host_name:
        config = TRAIN_CONFIGS['tacc']
    else:
        config = TRAIN_CONFIGS['local']
    eval(config)
