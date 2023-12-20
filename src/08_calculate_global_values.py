import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils.misc import scaling_df, read_result_file
import pdb
from pickle import dump
import matplotlib.pyplot as plt
from configs.configuration import H_MEAN, C_MEAN, H_STD, C_STD

cli_path = Path('new_cli/citydnn')

def read_climate_file(file_path):
    for i, file in enumerate(file_path.iterdir()):
        if i == 0:
            cli = pd.read_csv(file, sep='\t', skiprows=6)
        else:
            cli = pd.concat([cli, pd.read_csv(file, sep='\t', skiprows=6)])
    cli, scaler = scaling_df(cli)
    dump(scaler, open('data/citydnn_climate_scaler.pkl', 'wb'))
    pdb.set_trace()

def load_historgram(df, mode='cool'):
    if mode == 'cool':
        type_str = 'Cooling'
        c = '669bbc'
    else:
        type_str = 'Heating'
        c = '780000' 
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    values = df.values.flatten()
    axs[0].hist(values, bins=100, color='#'+c)
    axs[0].set_title(f'Histogram of all {type_str} loads')
    nonzero = values[(values != 0) & (~np.isnan(values))]
    print(f'Non-zero {type_str} loads: {nonzero.size}')
    axs[1].hist(nonzero, bins=100, color='#'+c)
    axs[1].set_title(f'Histogram of non-zero {type_str} loads')
    plt.tight_layout()
    plt.savefig(f'figs/{type_str}_hist.png')
    plt.close()
    return nonzero

def ratio_res_file_is_triggered(res_path):
    res_path = Path(res_path)
    for i, file in enumerate(res_path.iterdir()):
        if i == 0:
            res = read_result_file(file / f'{file.name}_TH.out')
        else:
            res = pd.concat([res, read_result_file(file/ f'{file.name}_TH.out')], axis=1)
    heat_df = res.loc[:, (res.columns.str.contains('Heating'))]
    heat_trigger_ratio = (heat_df != 0).sum().sum() / heat_df.size
    print(f'Heating trigger ratio: {heat_trigger_ratio:.4f}')
    cool_df = res.loc[:, (res.columns.str.contains('Cooling'))]
    cool_trigger_ratio = (cool_df != 0).sum().sum() / cool_df.size
    print(f'Cooling trigger ratio: {cool_trigger_ratio:.4f}')
    h_nonzero = load_historgram(heat_df, mode='heat')
    c_nonzero = load_historgram(cool_df)
    pdb.set_trace()

if __name__ == '__main__':
    # read_climate_file(cli_path)
    ratio_res_file_is_triggered('data/citydnn')