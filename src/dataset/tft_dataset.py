from dataset.dataset import CitySimTSDataset, CitySimDataModule
from utils.data_utils import FEAT_ORDER, DTYPE_MAP, DataTypes
import pickle
import numpy as np
import torch
from collections import OrderedDict

from utils.misc import *

FEAT_NAMES = ['s_cat', 's_cont', 'k_cat', 'k_cont', 'o_cat', 'o_cont', 'target', 'id']


class CitySimTFTDataset(CitySimTSDataset):
    def __init__(self,
                #  path, config,  # TFT config
                 # CitySim config
                 cli_df, res_df, bud_df, index, bud_key, in_len, out_len, transform=None
                 ):
        super().__init__(cli_df, res_df, bud_df, index, bud_key, in_len, out_len, transform)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        bud = self.bud_df.loc[self.index[idx, 0], self.bud_key].to_numpy() # (1, b_dim)
        cli = []
        for i in self.index[idx, 1]:
            cli.append(self.cli_df.iloc[i:i+self.in_len].to_numpy())
        cli = np.concatenate(cli, axis=0)  # (in_len, c_dim)
        res = []
        for i in self.index[idx]:
            # the last time step is the target
            res.append(self.res_df.iloc[i[1]:(i[1]+self.out_len),
                       (i[0]-5)*2:(i[0]-5)*2+2].to_numpy())
        res = np.concatenate(res, axis=0)  # (batch, out_len, 2)
        
        tensors = [
            torch.empty(0),
            torch.from_numpy(bud).float(),
            torch.empty(0),
            torch.from_numpy(cli).float(),
            torch.empty(0),
            torch.empty(0),
            torch.from_numpy(res).float(),
            torch.IntTensor(idx),
        ]
        return OrderedDict(zip(FEAT_NAMES, tensors))

class CitySimTFTDataModule(CitySimDataModule):
    def __init__(self, batch_size=64, cli_dir='./new_cli/citydnn', cli_loc='Portland_OR-hour', res_dir='./data/citydnn', building_path='./data/ut_building_info.csv', input_ts=1, output_ts=1):
        super().__init__(batch_size, cli_dir, cli_loc, res_dir, building_path, input_ts, output_ts)
    
    def setup(self, stage=None):
        cli_df = read_climate_file(self.cli_file)
        res_file = self.res_dir / self.cli_loc / f'{self.cli_loc}_TH.out'
        res_df = read_result_file(res_file)
        res_df = normalize_load(res_df, self.h_mean, self.h_std, self.c_mean, self.c_std)
        bud_df = read_building_info(self.building_path)
        bud_index = bud_df.id.to_numpy()
        cli_index = cli_df.index[:len(cli_df)-self.input_ts+1]
        index = np.array([np.tile(bud_index, len(cli_index)),
                         np.repeat(cli_index, len(bud_index))]).T
        generator1 = torch.Generator().manual_seed(1340)
        train_index, val_index, test_index = torch.utils.data.random_split(
            index, [0.64, 0.16, 0.2], generator=generator1)
        
        if stage == 'fit' or stage is None:
                self.train_dataset = CitySimTFTDataset(
                    cli_df, res_df, bud_df, train_index, self.bud_key, self.input_ts, self.output_ts, self.transform)
                self.val_dataset = CitySimTFTDataset(
                    cli_df, res_df, bud_df, val_index, self.bud_key, self.input_ts, self.output_ts, self.transform)
        if stage == 'test' or stage is None:
            self.test_dataset = CitySimTFTDataset(
                    cli_df, res_df, bud_df, test_index, self.bud_key, self.input_ts, self.output_ts, self.transform)
            