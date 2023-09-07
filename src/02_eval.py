import pdb
from model import Net, RNNNet
from dataset import CitySimDataModule
import lightning as L
import torch.nn as nn
import torch
from utils.vis import plot_heat_cool
torch.set_float32_matmul_precision('highest')
from pathlib import Path

input_seq_len = 1
dnn_model_path = 'lightning_logs/normalized_load_2/checkpoints/epoch=150-step=1508036.0-val_loss=0.07-v1.ckpt'
rnn_model_path = 'lightning_logs/ts_v0/checkpoints/epoch=287-step=2868767.0-val_loss=0.00-v1.ckpt'

fig_path = f'figs/dnn_input={input_seq_len}_output=0'

# model = RNNNet(input_dim=25, input_ts=24).load_from_checkpoint(rnn_model_path)
model = Net().load_from_checkpoint(dnn_model_path)
model.eval().to('cpu')


dm = CitySimDataModule(input_ts=input_seq_len)
dm.prepare_data()
dm.setup(stage='test')
heat_loss = 0
cool_loss = 0
fig_path = Path(fig_path)
fig_path.mkdir(parents=True, exist_ok=True)
for i, batch in enumerate(dm.test_dataloader()):
    x, y = model.split_reshape(batch)
    heat_hat, cool_hat = model(x)
    y[:, 0] = dm.heat_inverse_transform(y[:, 0])
    y[:, 1] = dm.cool_inverse_transform(y[:, 1])
    heat_hat = dm.heat_inverse_transform(heat_hat)
    cool_hat = dm.cool_inverse_transform(cool_hat)
    plot_heat_cool(y, heat_hat, cool_hat, fig_path/f'{i}.png')
    if i == 10:
        break

# heat_loss = heat_loss/len(dm.test_dataloader())
# cool_loss = cool_loss/len(dm.test_dataloader())
# total_loss = heat_loss + cool_loss
# print(f'heat_loss: {heat_loss}')
# print(f'cool_loss: {cool_loss}')
# print(f'total_loss: {total_loss}')
# trainer.predict(model, datamodule=dm)
