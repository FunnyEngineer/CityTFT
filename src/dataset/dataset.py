from torch.utils.data._utils.collate import default_collate
from typing import Optional, Union, Callable, Dict, Optional, Tuple, Type, Union
import torch.utils.data as data
import lightning as L
import torch
import numpy as np
import torchvision.transforms as transforms
import yaml
from pathlib import Path
from utils.misc import *
from utils.concat_input import combine_res_bud_cli
from collections import OrderedDict


class CitySimDataset(data.Dataset):
    def __init__(self, cli_df, res_df, bud_df, index, bud_key, transform=None):
        self.cli_df = cli_df
        self.res_df = res_df
        self.bud_df = bud_df
        self.bud_key = bud_key
        self.index = index.dataset[index.indices]
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        bud = self.bud_df.loc[self.index[idx, 0], self.bud_key].to_numpy()
        cli = self.cli_df.loc[self.index[idx, 1]].to_numpy()
        res = []
        for i in self.index[idx]:
            res = np.concatenate(
                [res, self.res_df.iloc[i[1], (i[0]-5)*2:(i[0]-5)*2+2].to_numpy()])
        res = torch.from_numpy(res.reshape(len(idx), -1))
        input = np.concatenate([bud, cli], axis=1)
        if self.transform:
            return self.transform(input), res  # return data, label (X, y)

        return input, res


class CitySimTSDataset(CitySimDataset):
    def __init__(self, cli_df, res_df, bud_df, index, bud_key, in_len,  out_len, mode='rnn', transform=None):
        super().__init__(cli_df, res_df, bud_df, index, bud_key, transform)
        self.in_len = in_len
        self.out_len = out_len
        self.mode = mode

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        bud = self.bud_df.loc[self.index[idx, 0],
                              self.bud_key].to_numpy()  # (batch, b_dim)
        bud = np.repeat(np.expand_dims(bud, axis=1),
                        self.in_len, axis=1)  # (batch, in_len, b_dim
        cli = []
        ts = []
        for i in self.index[idx, 1]:
            cli.append(self.cli_df.iloc[i:i+self.in_len].to_numpy())
            ts.append(np.arange(i, i+self.in_len) / 8760)
        cli = np.array(cli)  # (batch, in_len, c_dim)
        ts = np.array(ts)  # (batch, in_len, 1)

        res = []
        for i in self.index[idx]:
            # the last time step is the target
            res.append(self.res_df.iloc[i[1]:(i[1]+self.out_len),
                       (i[0]-5)*2:(i[0]-5)*2+2].to_numpy())
        res = np.array(res)  # (batch, out_len, 2)
        # (batch, in_len, b_dim+c_dim)
        input = np.concatenate([bud, cli], axis=2)
        if self.mode == 'atten':
            return self.transform(input), res, ts
        return self.transform(input), res


class CSTSMultiClimateDataset(data.Dataset):
    def __init__(self, cli_dir, res_dir, cli_locs, bud_path, bud_key, h_mean, h_std, c_mean, c_std,
                 seq_len=24, transform=None):
        self.cli_locs = cli_locs
        self.cli_df_list = {}
        self.res_df_list = {}

        n_cli_loc = len(cli_locs)
        self.n_cli_loc = n_cli_loc
        for cli_loc in cli_locs:
            self.cli_df_list[cli_loc] = read_climate_file(
                Path(cli_dir) / f'{cli_loc}.cli')
            self.res_df_list[cli_loc] = normalize_load(read_result_file(
                res_dir / cli_loc / f'{cli_loc}_TH.out'), h_mean, h_std, c_mean, c_std)

        self.bud_df = read_building_info(bud_path)[bud_key]
        self.n_bud = len(self.bud_df)
        self.bud_ind = self.bud_df.index.to_numpy()

        self.seq_len = seq_len
        self.step = 24
        self.n_cli_sam = (8760 - seq_len) // self.step + 1  # default step = 1
        self.transform = transform

    def __len__(self):
        return self.n_cli_sam * self.n_bud * self.n_cli_loc

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        input = []
        res = []
        for i in idx:
            loc_i = i // (self.n_bud * self.n_cli_sam)
            bud_i = (i % (self.n_bud * self.n_cli_sam) % self.n_bud) + 5
            time_i = (i % (self.n_bud * self.n_cli_sam) //
                      self.n_bud) * self.step
            sin_bud = self.bud_df.loc[bud_i].to_numpy()
            sin_bud = np.repeat(np.expand_dims(sin_bud, axis=0),
                                self.seq_len, axis=0)  # (in_len, b_dim)
            sin_cli = self.cli_df_list[self.cli_locs[loc_i]
                                       ].iloc[time_i:time_i+self.seq_len].to_numpy()  # (in_len, c_dim)
            sin_ts = np.expand_dims(np.arange(time_i, time_i+self.seq_len) /
                                    8760, axis=1)  # (in_len, 1)
            sin_res = self.res_df_list[self.cli_locs[loc_i]].iloc[time_i:time_i +
                                                                  self.seq_len, (bud_i-5)*2: (bud_i-5)*2+2].to_numpy()  # (in_len, 2)
            if np.isnan(sin_res).any():
                continue
            sin_input = np.concatenate([sin_bud, sin_cli, sin_ts],
                                       axis=1)  # (in_len, b_dim+c_dim+1)
            input.append(sin_input)
            res.append(sin_res)

        input = torch.from_numpy(np.array(input))
        res = torch.from_numpy(np.array(res))
        return input, res


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    inp = []
    label = []
    for s_inp, s_label in batch:
        if s_inp.nelement() == 0:
            continue
        inp.append(s_inp)
        label.append(s_label)
    return torch.stack(inp, 0), torch.stack(label, 0)


FEAT_NAMES = ['s_cat', 's_cont', 'k_cat',
              'k_cont', 'o_cat', 'o_cont', 'target', 'id']


class CSTSMultiClimateTFTDataset(CSTSMultiClimateDataset):
    def __init__(self, cli_dir, res_dir, cli_locs, bud_path, bud_key, seq_len=24, transform=None):
        super().__init__(cli_dir, res_dir, cli_locs, bud_path, bud_key, seq_len, transform)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        bud = []
        cli = []
        res = []
        for i in idx:
            loc_i = i // (self.n_bud * self.n_cli_sam)
            bud_i = (i % (self.n_bud * self.n_cli_sam) % self.n_bud) + 5
            time_i = (i % (self.n_bud * self.n_cli_sam) //
                      self.n_bud) * self.step
            sin_bud = self.bud_df.loc[bud_i].to_numpy()  # (1, b_dim)
            sin_cli = self.cli_df_list[self.cli_locs[loc_i]
                                       ].iloc[time_i:time_i+self.seq_len].to_numpy()  # (in_len, c_dim)

            sin_res = self.res_df_list[self.cli_locs[loc_i]].iloc[time_i:time_i +
                                                                  self.seq_len, (bud_i-5)*2: (bud_i-5)*2+2].to_numpy()  # (in_len, 2)
            if np.isnan(sin_res).any():
                continue

            bud.append(sin_bud)
            cli.append(sin_cli)
            res.append(sin_res)
        bud = np.array(bud)  # (batch, 1, b_dim)
        cli = np.concatenate(cli, axis=0) if cli else np.array(
            cli)  # (in_len, c_dim)
        res = np.concatenate(res, axis=0) if res else np.array(
            res)  # (batch, out_len, 2)
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


class CSTSMCMUDataset(CSTSMultiClimateDataset):
    def __init__(self, cli_dir, res_dir, cli_locs, bud_path, bud_key, seq_len=24, transform=None):
        self.cli_locs = cli_locs
        self.cli_df_list = {}
        self.res_df_list = {}

        n_cli_loc = len(cli_locs)
        self.n_cli_loc = n_cli_loc

        bud_path = Path(bud_path)
        bud_list = list(bud_path.glob('*.csv'))
        self.n_bud = 0
        self.df = pd.DataFrame()
        for bud_path in bud_list:
            for cli_loc in cli_locs:
                print(f'Processing {bud_path.stem} {cli_loc}')
                res_path = res_dir / bud_path.stem / f'{cli_loc}.csv'
                cli_path = cli_dir / f'{cli_loc}.cli'
                self.df = pd.concat(
                    [self.df, combine_res_bud_cli(cli_path, res_path, bud_path)])
        self.df = self.df.reset_index(drop=True)

        self.seq_len = seq_len
        self.step = 24
        self.n_cli_sam = (8760 - seq_len) // self.step + 1  # default step = 1
        self.transform = transform

        # calculate the number of samples
        self.n_samples = n_cli_loc * self.n_bud * self.n_cli_sam

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        return self.transform(self.df.iloc[idx, :-2].to_numpy()), self.df.iloc[idx, -2:].to_numpy()


def collate_tft_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    # filter out empty value before collate:
    batch = [x for x in batch if x['s_cont'].nelement() != 0]
    return default_collate(batch)


class CitySimDataModule(L.LightningDataModule):
    """
    CitySimDataModule
    """

    def __init__(self, scaling, batch_size=64, cli_dir='./new_cli/citydnn',
                 cli_split_file='src/climate_split.yaml', res_dir='./data/citydnn',
                 building_path='./data/ut_building_info.csv', input_ts=1, output_ts=1, mode='rnn'):
        super().__init__()
        self.scaling = scaling
        self.batch_size = batch_size
        # self.cli_file = Path(cli_dir) / f'{cli_loc}.cli'
        self.cli_dir = Path(cli_dir)
        self.res_dir = Path(res_dir)
        self.building_path = building_path
        self.bud_key = yaml.load(open('src/input_vars.yaml', 'r'),
                                 Loader=yaml.FullLoader)['BUD_PROPS']

        # prepare climate location split
        cli_split = yaml.load(open(cli_split_file, 'r'),
                              Loader=yaml.FullLoader)
        self.train_cli_locs = cli_split['TRAIN']
        self.val_cli_locs = cli_split['VAL']
        self.test_cli_locs = cli_split['TEST']

        self.mode = mode

        self.heat_key = 'Heating(Wh)'
        self.cool_key = 'Cooling(Wh)'
        self.input_ts = input_ts  # if 1, direct prediction, if >1, time series prediction
        self.output_ts = output_ts  # if 1, means predict the same time step
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.collate_fn = collate_tensor_fn if self.mode != 'tft' else collate_tft_fn

    def prepare_data(self):
        pass

    def heat_inverse_transform(self, heat):
        return heat * self.scaling.H_STD + self.scaling.H_MEAN

    def cool_inverse_transform(self, cool):
        return cool * self.scaling.C_STD + self.scaling.C_MEAN

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.mode != 'tft':
                self.train_dataset = CSTSMultiClimateDataset(
                    self.cli_dir, self.res_dir, self.train_cli_locs, self.building_path, self.bud_key, transform=self.transform)
                self.val_dataset = CSTSMultiClimateDataset(
                    self.cli_dir, self.res_dir, self.val_cli_locs, self.building_path, self.bud_key, transform=self.transform)
            else:
                self.train_dataset = CSTSMultiClimateTFTDataset(
                    self.cli_dir, self.res_dir, self.train_cli_locs, self.building_path, self.bud_key, transform=self.transform)
                self.val_dataset = CSTSMultiClimateTFTDataset(
                    self.cli_dir, self.res_dir, self.val_cli_locs, self.building_path, self.bud_key, transform=self.transform)
            # if self.input_ts > 1:
            #     self.train_dataset = CitySimTSDataset(
            #         cli_df, res_df, bud_df, train_index, self.bud_key, self.input_ts, self.output_ts, self.mode, self.transform)
            #     self.val_dataset = CitySimTSDataset(
            #         cli_df, res_df, bud_df, val_index, self.bud_key, self.input_ts, self.output_ts, self.mode, self.transform)
            # else:
            #     self.train_dataset = CitySimDataset(
            #         cli_df, res_df, bud_df, train_index, self.bud_key, self.transform)
            #     self.val_dataset = CitySimDataset(
            #         cli_df, res_df, bud_df, val_index, self.bud_key, self.transform)
        if stage == 'test' or stage is None:

            if self.mode != 'tft':
                self.test_dataset = CSTSMultiClimateDataset(
                    self.cli_dir, self.res_dir, self.test_cli_locs, self.building_path, self.bud_key, transform=self.transform)
            else:
                self.test_dataset = CSTSMultiClimateTFTDataset(
                    self.cli_dir, self.res_dir, self.test_cli_locs, self.building_path, self.bud_key, transform=self.transform)

            # if self.input_ts > 1:
            #     self.test_dataset = CitySimTSDataset(
            #         cli_df, res_df, bud_df, test_index, self.bud_key, self.input_ts, self.output_ts, self.mode, self.transform)
            # else:
            #     self.test_dataset = CitySimDataset(
            #         cli_df, res_df, bud_df, test_index, self.bud_key, self.transform)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=20)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=20)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=20)

    def predict_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=20)


class MultiUrbanCitySim(CitySimDataModule):
    def __init__(self, batch_size=64, cli_dir='./new_cli/citydnn',
                 cli_split_file='src/climate_split.yaml', res_dir='./data/random_urban/export_csv',
                 building_path='./data/random_urban', input_ts=24, output_ts=24, mode='rnn'):
        super().__init__(batch_size, cli_dir, cli_split_file,
                         res_dir, building_path, input_ts, output_ts, mode)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CSTSMCMUDataset(
                self.cli_dir, self.res_dir, self.train_cli_locs, self.building_path, self.bud_key, transform=self.transform)
            self.val_dataset = CSTSMCMUDataset(
                self.cli_dir, self.res_dir, self.val_cli_locs, self.building_path, self.bud_key, transform=self.transform)
        if stage == 'test' or stage is None:
            self.test_dataset = CSTSMCMUDataset(
                self.cli_dir, self.res_dir, self.test_cli_locs, self.building_path, self.bud_key, transform=self.transform)
