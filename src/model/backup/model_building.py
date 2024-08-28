from typing import Any
import lightning as L
import torch
import torch.nn as nn
import pdb

class BudNet(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(25, 128), nn.BatchNorm1d(
            128), nn.ReLU(), nn.Linear(128, 256), nn.BatchNorm1d(256))
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x) -> Any:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        bud_env, target_bud, label = batch
        pdb.set_trace()
        pred_bud = self(bud_env)
        loss = nn.MSELoss()(pred_bud, target_bud)
        self.log('train/total_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)