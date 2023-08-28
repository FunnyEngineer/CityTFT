import lightning as L
import torch
import torch.nn as nn
import pdb

class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(25, 128), nn.BatchNorm1d(
            128), nn.ReLU(), nn.Linear(128, 256), nn.BatchNorm1d(256))
        self.heat_decoder = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).float()
        y = y.view(y.size(0), -1).float()
        z = self.encoder(x)
        heat_hat = self.heat_decoder(z)
        cool_hat = self.cool_decoder(z)

        heat_loss = nn.functional.mse_loss(heat_hat, y[:, 0].unsqueeze(1))
        cool_loss = nn.functional.mse_loss(cool_hat, y[:, 1].unsqueeze(1))
        loss = heat_loss + cool_loss
        self.log_dict({"train/heat_loss": heat_loss,
                      "train/cool_loss": cool_loss, "train/total_loss": loss, "global_step": self.global_step})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).float()
        y = y.view(y.size(0), -1).float()
        z = self.encoder(x)
        heat_hat = self.heat_decoder(z)
        cool_hat = self.cool_decoder(z)

        heat_loss = nn.functional.mse_loss(heat_hat, y[:, 0].unsqueeze(1))
        cool_loss = nn.functional.mse_loss(cool_hat, y[:, 1].unsqueeze(1))
        loss = heat_loss + cool_loss
        self.log_dict({"val/heat_loss": heat_loss,
                      "val/cool_loss": cool_loss, "val/total_loss": loss})
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1).float()
        y = y.view(y.size(0), -1).float()
        z = self.encoder(x)
        heat_hat = self.heat_decoder(z)
        cool_hat = self.cool_decoder(z)

        heat_loss = nn.functional.mse_loss(heat_hat, y[:, 0].unsqueeze(1))
        cool_loss = nn.functional.mse_loss(cool_hat, y[:, 1].unsqueeze(1))
        loss = heat_loss + cool_loss
        self.log_dict({"test/heat_loss": heat_loss,
                      "test/cool_loss": cool_loss, "test/total_loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
