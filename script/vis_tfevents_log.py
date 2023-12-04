import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pdb
import pickle
from tensorflow.python.summary.summary_iterator import summary_iterator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab

# plt.style.use('seaborn-v0_8')
# plt.style.use('seaborn')
# plt.rcParams.update({'axes.facecolor': 'white'})
# plt.figure(figsize=(15, 7))
plt.style.use('./script/poster.mplstyle')
result_dict = {
    'DNN_P2P': 'lightning_logs/normalized_load_2/events.out.tfevents.1693186118.FunnyEngineer.633790.0',
    'DNN_Seq2P': 'lightning_logs/trans_with_ts_v2_adamw_lr_1e-5/events.out.tfevents.1694839908.FunnyEngineer.51895.0',
    'RNN_Seq2P': 'lightning_logs/ts_v0/events.out.tfevents.1693366689.FunnyEngineer.63752.0',
    'RNN_Seq2Seq': 'lightning_logs/rnn_seq_v0/events.out.tfevents.1693940324.FunnyEngineer.54863.0',
    'Transformer_Seq2Seq': 'lightning_logs/rnn_seq_with_embed_v2/events.out.tfevents.1694927832.FunnyEngineer.476637.0'
}
df_dict = {}
def tfevent_to_df():
    for k, v in result_dict.items():
        print(f'Processing {k}')
        tags = ["train/total_loss", "epoch", "val/total_loss"]
        df = pd.DataFrame(columns=tags)
        for e in summary_iterator(v):
            if e.summary.value == []:
                continue
            summary = e.summary.value[0]
            if summary.tag in tags:
                df.loc[e.step, summary.tag] = summary.simple_value
        df_dict[k] = df
    pickle.dump(df_dict, open('./data/log_df.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def plot_training_loss():
    df_dict = pickle.load(open('./data/log_df.pkl', 'rb'))
    fig, axs = plt.subplots(1, 1, figsize=(15, 7))
    for k, v in df_dict.items():
        if k == 'RNN_Sep2Seq':
            k = 'RNN_Seq2Seq'
        v = v.groupby('epoch').mean()
        v = v.groupby(np.arange(len(v))//5).mean()
        axs.plot((v.index + 0.5) * 5, v.iloc[:, 0], label=k)
        # axs[1].plot((v.index + 0.5) * 5, v.iloc[:, 1], label=k)
        
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    # plt.legend()
    plt.xlabel('# of Epoch')
    axs.set_yscale('log')
    # axs[0].set_ylim([0.0001, 0.1])
    # axs[1].set_yscale('log')
    axs.set_ylabel('Training MSE')
    # axs[1].set_ylabel('Validation MSE')
    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE) 
    plt.legend()
    plt.show()
    # plt.savefig('figs/train_loss.png', bbox_inches='tight')
    
if __name__ == '__main__':
    plot_training_loss()
