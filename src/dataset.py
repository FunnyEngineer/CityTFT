import torch.utils.data as data
import lightning as L
import torch
import numpy as np
import torchvision.transforms as transforms
import yaml
from pathlib import Path
from utils.misc import *

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
            res = np.concatenate([res, self.res_df.iloc[i[1], (i[0]-5)*2:(i[0]-5)*2+2].to_numpy()])
        res = torch.from_numpy(res.reshape(len(idx), -1))
        input = np.concatenate([bud, cli], axis=1)
        if self.transform:
            return self.transform(input), res  # return data, label (X, y)

        return input, res


class CitySimDataModule(L.LightningDataModule):
    """
    CitySimDataModule
    """

    def __init__(self, batch_size=64, cli_dir='./new_cli/citydnn',
                 cli_loc='Portland_OR-hour', res_dir='./data/citydnn',
                 building_path='./data/ut_building_info.csv'):
        super().__init__()
        self.batch_size = batch_size
        self.cli_file = Path(cli_dir) / f'{cli_loc}.cli'
        self.cli_loc = cli_loc
        self.res_dir = Path(res_dir)
        self.building_path = building_path
        self.bud_key = yaml.load(open('src/input_vars.yaml', 'r'),
                                 Loader=yaml.FullLoader)['BUD_PROPS']

        self.heat_key = 'Heating(Wh)'
        self.cool_key = 'Cooling(Wh)'
        self.h_mean = 124264.10207582312
        self.h_std = 209963.99301021238
        self.c_mean = -40422.87019045903
        self.c_std = 127541.16314976662

        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        cli_df = read_climate_file(self.cli_file)
        res_file = self.res_dir / self.cli_loc / f'{self.cli_loc}_TH.out'
        res_df = read_result_file(res_file)
        res_df = normalize_load(res_df, self.h_mean, self.h_std, self.c_mean, self.c_std)
        bud_df = read_building_info(self.building_path)
        bud_index = bud_df.id.to_numpy()
        cli_index = cli_df.index
        index = np.array([np.tile(bud_index, len(cli_index)),
                         np.repeat(cli_index, len(bud_index))]).T
        generator1 = torch.Generator().manual_seed(1340)
        train_index, val_index, test_index = torch.utils.data.random_split(
            index, [0.64, 0.16, 0.2], generator=generator1)

        if stage == 'fit' or stage is None:
            self.train_dataset = CitySimDataset(
                cli_df, res_df, bud_df, train_index, self.bud_key, self.transform)
            self.val_dataset = CitySimDataset(
                cli_df, res_df, bud_df, val_index, self.bud_key, self.transform)
        if stage == 'test' or stage is None:
            self.test_dataset = CitySimDataset(
                cli_df, res_df, bud_df, test_index, self.bud_key, self.transform)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                               num_workers=20)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                               num_workers=20)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                               num_workers=20)

    def predict_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                               num_workers=20)
