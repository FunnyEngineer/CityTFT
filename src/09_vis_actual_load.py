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
    fig = plt.figure(constrained_layout=True, figsize=(20, 8))
    figs = fig.subfigures(2, 1)
    for row, subfig in enumerate(figs):
        plot_true = False
        # set global font size larger
        plt.rcParams.update({'font.size': 24})
        subfig.suptitle(['Heating loads', 'Cooling loads'][row])

        target_len = 1152

        axs = subfig.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [4, 1]}, sharey=True)
        true_c = ['red', 'blue'][row]
        long_si = [450, 1050]
        short_si = 735
        for mode in pred_dict:
            predictions = pred_dict[mode]
            heat_true = torch.zeros(target_len)
            heat_pred = torch.zeros(target_len)
            i = 0
            bi = start_bi
            while target_len > i:
                batch_len = predictions[bi][1][:,:, 0].nelement()
                if (target_len-i) >= batch_len:
                    heat_true[i:i+batch_len] = predictions[bi][1][:, :, row].flatten()
                    heat_pred[i:i+batch_len] = predictions[bi][0][row].flatten()
                else:
                    heat_true[i:] = predictions[bi][1][:, :, row].flatten()[:(target_len-i)]
                    heat_pred[i:] = predictions[bi][0][row].flatten()[:(target_len-i)]
                i += batch_len
                bi += 1
            if plot_true == False:
                axs[0].plot(range(long_si[0], long_si[1]), heat_true[long_si[0]:long_si[1]].cpu(), color=true_c, label='CitySim', lw=2)
                axs[1].plot(range(short_si, short_si+72), heat_true[short_si:short_si+72].cpu(), color=true_c, label='CitySim', lw=2)

                plot_true = True
            axs[0].plot(range(long_si[0], long_si[1]), heat_pred[long_si[0]:long_si[1]].cpu(), color=cool_dict[mode], label=f'{mode}', ls='-.')
            axs[1].plot(range(short_si, short_si+72), heat_pred[short_si:short_si+72].cpu(), color=cool_dict[mode], label=f'{mode}', ls='-.')
            # limit x range to 0 and length of target
            axs[0].set_xlim(long_si[0], long_si[1])
            axs[1].set_xlim(short_si, short_si+72)
        # legend with smaller font size
        axs[0].legend(prop={'size': 18})
    plt.savefig('figs/demonstration_overall.png')


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
