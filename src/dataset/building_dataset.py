from typing import Optional
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import torch.utils.data as data
import lightning as L
import torch
import torchvision.transforms as transforms
import yaml
import numpy as np
from utils.misc import *


class BuildingClsDataset(data.Dataset):
    def __init__(self, bud_df, bud_key) -> None:
        super().__init__()
        self.bud_df = bud_df
        self.bud_key = bud_key
        self.index = bud_df.id.to_numpy()
        pdb.set_trace()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx) == int:
            idx = [idx]

        bud_env = self.bud_df.loc[:, self.bud_key].to_numpy()
        label = self.index[idx]
        target_bud = self.bud_df.loc[label, self.bud_key].to_numpy()
        return bud_env, target_bud, label


class BuildingClsModule(L.LightningDataModule):
    """
    Building classification module
    Used for data loading and preprocessing
    """

    def __init__(self, batch_size=64, building_path='./data/ut_building_info.csv') -> None:
        super().__init__()
        self.bs = batch_size
        self.building_path = building_path
        self.bud_key = yaml.load(open('src/input_vars.yaml', 'r'),
                                 Loader=yaml.FullLoader)['BUD_PROPS']

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        bud_df = read_building_info(self.building_path)
        bud_index = bud_df.id.to_numpy()
        np.random.shuffle(bud_index)
        train_index, val_index = bud_index[:int(len(bud_index) * 0.8)], bud_index[int(len(bud_index) * 0.8):]
        train_df = bud_df.loc[train_index]
        val_df = bud_df.loc[val_index]
        if stage == 'fit' or stage is None:
            self.train_dataset = BuildingClsDataset(train_df, self.bud_key)
            self.val_dataset = BuildingClsDataset(val_df, self.bud_key)

        if stage == 'test' or stage is None:
            self.test_dataset = BuildingClsDataset(val_df, self.bud_key)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return data.DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True, num_workers=20)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.val_dataset, batch_size=self.bs, shuffle=False, num_workers=20)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False, num_workers=20)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return data.DataLoader(self.test_dataset, batch_size=self.bs, shuffle=False, num_workers=20)
