from dataset.dataset import CitySimDataModule
import lightning as L
import torch.nn as nn
import torch
from utils.vis import plot_heat_cool
from model.rnn import RNNSeqNetV2
from model.transformer import TransNetV2
import pdb
import matplotlib.pyplot as plt
import pickle

input_dim = 26
input_seq_len = 24

cool_dict = {'RNN': '#513b56', 'Transformer': '#f4a259', 'TFT': '#25a18e'} # '#F4D35E', '#EE964B', '#F95738'
def plot_heat_cool_seq_batch(pred_dict, start_bi=0):
    plot_true = False
    fig, axs = plt.subplots(2, 1, figsize=(20, 6))
    for mode in pred_dict:
        predictions = pred_dict[mode]
        target_len = 1536
        heat_true = torch.zeros(target_len)
        heat_pred = torch.zeros(target_len)
        cool_true = torch.zeros(target_len)
        cool_pred = torch.zeros(target_len)
        i = 0
        bi = start_bi
        while target_len > i:
            batch_len = predictions[bi][1][:,:, 0].nelement()
            if (target_len-i) >= batch_len:
                heat_true[i:i+batch_len] = predictions[bi][1][:, :, 0].flatten()
                heat_pred[i:i+batch_len] = predictions[bi][0][0].flatten()
                cool_true[i:i+batch_len] = predictions[bi][1][:, :, 1].flatten()
                cool_pred[i:i+batch_len] = predictions[bi][0][1].flatten()
            else:
                heat_true[i:] = predictions[bi][1][:, :, 0].flatten()[:(target_len-i)]
                heat_pred[i:] = predictions[bi][0][0].flatten()[:(target_len-i)]
                cool_true[i:] = predictions[bi][1][:, :, 1].flatten()[:(target_len-i)]
                cool_pred[i:] = predictions[bi][0][1].flatten()[:(target_len-i)]
            i += batch_len
            bi += 1
        if plot_true == False:
            axs[0].plot(heat_true.cpu(), color='red', label='CitySim', lw=3)
            axs[1].plot(cool_true.cpu(), color='blue', label='CitySim', lw=3)
            plot_true = True
        axs[0].plot(heat_pred.cpu(), color=cool_dict[mode], label=f'{mode}', ls='-.')
        axs[1].plot(cool_pred.cpu(), color=cool_dict[mode], label=f'{mode}', ls='-.')
    axs[0].set_title('Heating loads')
    axs[1].set_title('Cooling loads')
    axs[0].legend()
    axs[1].legend()
    plt.show()


def visual_load_differences():

    # load the model
    # rnn = RNNSeqNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len).load_from_checkpoint(
    #     # 'multi_cli/rnn_with_prob_quantile_v3_adamW_lr1e-4/checkpoints/epoch=00-step=9103.0546875-val_loss=0.85979670-v2.ckpt') # best
    #     'multi_cli/rnn_with_prob_quantile_v3_adamW_lr1e-4/checkpoints/epoch=398-step=3632125.25-val_loss=2.11352944.ckpt') # last
    # trans = TransNetV2(input_dim=input_dim, input_ts=input_seq_len, output_ts=input_seq_len).load_from_checkpoint(
    #     # 'multi_cli/trans_v3_with_prob_quantile_adamW_lr1e-4/checkpoints/epoch=01-step=18206.109375-val_loss=0.81011742-v1.ckpt') # best
    #     'multi_cli/trans_v3_with_prob_quantile_adamW_lr1e-4/checkpoints/last.ckpt') # last
    # trans.eval()
    # dm = CitySimDataModule(input_ts=input_seq_len, output_ts=input_seq_len)
    # dm.setup(stage='test')

    # # test the model
    # trainer = L.Trainer()
    # trans_pred = trainer.predict(trans, dm.test_dataloader())
    # rnn_pred = trainer.predict(rnn, dm.test_dataloader())
    rnn_pred = pickle.load(open('predictions/rnn_last.pkl', 'rb'))
    trans_pred = pickle.load(open('predictions/trans_last.pkl', 'rb'))
    tft_pred = pickle.load(open('predictions/tft_last.pkl', 'rb'))
    plot_heat_cool_seq_batch({'RNN': rnn_pred, 'Transformer': trans_pred, 'TFT': tft_pred}, 1520)
    pdb.set_trace()


if __name__ == '__main__':
    visual_load_differences()
