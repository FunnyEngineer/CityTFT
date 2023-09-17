from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pdb
from model.model import Net, RNNSeqNet, TransformerSeqNet, TransNetV2
from dataset.dataset import CitySimDataModule
import lightning as L
import torch.nn as nn
import torch
from utils.vis import plot_heat_cool, plot_heat_cool_seq_batch
torch.set_float32_matmul_precision('highest')

input_seq_len = 1

input_dim = 25
input_seq_len = 24

dnn_model_path = 'lightning_logs/normalized_load_2/checkpoints/epoch=150-step=1508036.0-val_loss=0.07-v1.ckpt'
rnn_model_path = 'lightning_logs/rnn_seq_v0/checkpoints/epoch=314-step=3137714.0-val_loss=0.00-v1.ckpt'
trans_model_path = 'lightning_logs/trans_with_ts_v2_adamw_lr_1e-5/checkpoints/epoch=382-step=3815062.0-val_loss=0.01506070-v1.ckpt'

fig_path = Path(f'figs/dnn_rnn_trans_input={input_seq_len}_output={input_seq_len}')

model2 = RNNSeqNet(input_dim=25, input_ts=24, output_ts=24).load_from_checkpoint(rnn_model_path)
model = Net().load_from_checkpoint(dnn_model_path)
model3 = TransNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len)
model.eval().to('cpu')
model2.eval().to('cpu')
model3.eval().to('cpu')

dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len, mode='atten')
dm.prepare_data()
dm.setup(stage='test')
heat_pred = {}
cool_pred = {}
for i, batch in enumerate(dm.test_dataloader()):
    x, y = model3.split_reshape(batch)
    heat_hat, cool_hat = model3(x)
    heat_pred['atten'] = dm.heat_inverse_transform(heat_hat)
    cool_pred['atten'] = dm.cool_inverse_transform(cool_hat)

    x,y = model2.split_reshape(batch[:-1])
    heat_rnn, cool_rnn = model2(x)
    y[:, :, 0] = dm.heat_inverse_transform(y[:, :, 0])
    y[:, :, 1] = dm.cool_inverse_transform(y[:, :, 1])

    heat_pred['rnn'] = dm.heat_inverse_transform(heat_rnn)
    cool_pred['rnn'] = dm.cool_inverse_transform(cool_rnn)
    plot_heat_cool_seq_batch(y, heat_pred, cool_pred, fig_path / f'{i}.png')
    if i > 5:
        break



pdb.set_trace()