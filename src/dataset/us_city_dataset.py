from typing import Optional, Union, Callable, Dict, Optional, Tuple, Type, Union
import torch.utils.data as data
import lightning as L
import torch
import numpy as np
import torchvision.transforms as transforms
import yaml
from pathlib import Path
from utils.misc import *
from collections import OrderedDict
from torch.utils.data._utils.collate import default_collate


# global mean and std
h_mean = 208067.33730670897
h_std = 241361.16853850652
c_mean = -227953.4075332571
c_std = 259908.86350470388


def collate_tft_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    # filter out empty value before collate:
    batch = [x for x in batch if x['s_cont'].nelement() != 0]
    return default_collate(batch)


def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    inp = []
    label = []
    for s_inp, s_label in batch:
        if s_inp.nelement() == 0 or s_label.nelement() == 0:
            continue
        inp.append(s_inp)
        label.append(s_label)
    return torch.stack(inp, 0), torch.stack(label, 0)


class USCity(data.Dataset):
    def __init__(self, ref_csv, bud_dir, cli_dir, res_dir, input_ts, bud_key) -> None:
        super().__init__()
        self.cli_data = OrderedDict()
        self.bud_data = OrderedDict()
        self.res_data = OrderedDict()
        self.index_buds = np.array([], dtype=np.int64)
        self.index_city = np.array([], dtype=np.int64)
        self.bud_key = bud_key

        self.time_step = 24
        self.ts_length = input_ts
        self.total_length = 8760
        self.n_timestamp = (
            (self.total_length - self.ts_length) // self.time_step + 1)

        self.setup(ref_csv, bud_dir, cli_dir, res_dir)

    def setup(self, ref_csv, bud_dir, cli_dir, res_dir):
        ref = pd.read_csv(ref_csv)
        ref = ref.sort_values(by=['city'], ignore_index=True)
        # record the number of cities
        # record the number of buildings in each city
        for i, row in ref.iterrows():
            cli_file_name = row['climate']
            cli_path = list(cli_dir.glob(f'**/{cli_file_name}'))[0]
            cli_df = read_climate_file(cli_path)

            bud_file_name = row['bud']
            bud_path = list(bud_dir.glob(f'**/{bud_file_name}'))[0]
            bud_df = read_building_info(bud_path)

            res_file_name = row['result']
            res_path = list(res_dir.glob(f'**/{res_file_name}'))[0]
            res_df = read_result_file(res_path)

            # normalize the load
            res_df = normalize_load(res_df, h_mean, h_std, c_mean, c_std)
            self.index_buds = np.append(self.index_buds, range(len(bud_df)))
            self.index_city = np.append(
                self.index_city, np.repeat(i, len(bud_df)))

            # add cli, bud and res to self.data
            self.cli_data[i] = cli_df
            self.bud_data[i] = bud_df
            self.res_data[i] = res_df

    def __len__(self):
        return len(self.index_buds) * self.n_timestamp

    def __getitem__(self, index):
        quo = index // self.n_timestamp
        rem = index % self.n_timestamp
        city = self.index_city[quo]
        bud_i = self.index_buds[quo]

        bud_var = self.bud_data[city].loc[bud_i, self.bud_key].values
        bud_var = np.repeat(np.expand_dims(
            bud_var, axis=0), self.ts_length, axis=0)

        cli_var = self.cli_data[city].iloc[rem:rem+self.ts_length].values
        ts_var = np.expand_dims(
            np.arange(rem, rem+self.ts_length) / 8760, axis=1)  # (in_len, 1)

        input = np.concatenate([bud_var, cli_var, ts_var], axis=1)
        res_var = self.res_data[city].iloc[rem:rem +
                                           self.ts_length, (bud_i*2):(bud_i*2 + 2)].values

        input = torch.from_numpy(input)
        res_var = torch.from_numpy(res_var)
        return input, res_var


# data Module
class USCityDataModule(L.LightningDataModule):
    """
    USCityDataModule
    """

    def __init__(self, input_ts, batch_size=64, cli_dir='../US_cities/climate/historic', res_dir='../US_cities/result',
                 bud_dir='../US_cities/bud', ref_csv='../US_cities/ref.csv', mode='rnn'):
        super().__init__()
        self.intput_ts = input_ts
        self.batch_size = batch_size
        self.ref_csv = ref_csv
        self.cli_dir = Path(cli_dir)
        self.res_dir = Path(res_dir)
        self.bud_dir = Path(bud_dir)
        self.bud_key = yaml.load(open('src/input_vars.yaml', 'r'),
                                 Loader=yaml.FullLoader)['BUD_PROPS']

        self.mode = mode

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.collate_fn = collate_tensor_fn if self.mode != 'tft' else collate_tft_fn

    def prepare_data(self):
        pass

    def heat_inverse_transform(self, heat):
        return heat * h_std + h_mean

    def cool_inverse_transform(self, cool):
        return cool * c_std + c_mean

    def setup(self, stage=None):
        dataset = USCity(self.ref_csv, self.bud_dir,
                         self.cli_dir, self.res_dir, self.intput_ts, self.bud_key)
        # use dataset to split train, val and test
        self.train, self.val, self.test = data.random_split(
            dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(1340)
        )

    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=16)

    def val_dataloader(self):
        return data.DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn)
