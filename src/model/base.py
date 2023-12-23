import lightning as L
import torch
import torch.nn as nn
import pdb


class Net(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(ignore=["scaling"])
        self.encoder = nn.Sequential(nn.Linear(25, 128), nn.BatchNorm1d(
            128), nn.ReLU(), nn.Linear(128, 256), nn.BatchNorm1d(256))
        self.heat_decoder = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))

    def split_reshape(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1).float()
        y = y.view(y.size(0), -1).float()
        return x, y

    def forward(self, x):
        z = self.encoder(x)
        heat_hat = self.heat_decoder(z)
        cool_hat = self.cool_decoder(z)
        return heat_hat, cool_hat

    def criterion(self, heat_hat, cool_hat, y):
        heat_loss = nn.functional.mse_loss(heat_hat, y[:, 0].unsqueeze(1))
        cool_loss = nn.functional.mse_loss(cool_hat, y[:, 1].unsqueeze(1))
        return heat_loss, cool_loss

    def training_step(self, batch, batch_idx):
        x, y = self.split_reshape(batch)

        heat_hat, cool_hat = self(x)

        heat_loss, cool_loss = self.criterion(heat_hat, cool_hat, y)
        loss = heat_loss + cool_loss
        self.log_dict({"train/heat_loss": heat_loss, "train/cool_loss": cool_loss,
                       "train/total_loss": loss, "global_step": self.global_step})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.split_reshape(batch)

        heat_hat, cool_hat = self(x)

        heat_loss, cool_loss = self.criterion(heat_hat, cool_hat, y)
        loss = heat_loss + cool_loss
        self.log_dict({"val/heat_loss": heat_loss,
                      "val/cool_loss": cool_loss, "val/total_loss": loss})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = self.split_reshape(batch)

        heat_hat, cool_hat = self(x)

        heat_loss, cool_loss = self.criterion(heat_hat, cool_hat, y)
        loss = heat_loss + cool_loss
        self.log_dict({"test/heat_loss": heat_loss,
                      "test/cool_loss": cool_loss, "test/total_loss": loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class RNNNet(Net):
    def __init__(self, input_dim, input_ts, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.input_ts = input_ts
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=hidden_dim,
                               num_layers=2,
                               batch_first=True)
        self.heat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.BatchNorm1d(hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, x):
        z, _ = self.encoder(x)
        z = z[:, -1, :]
        heat_hat = self.heat_decoder(z)
        cool_hat = self.cool_decoder(z)
        return heat_hat, cool_hat

    def split_reshape(self, batch):
        x, y = batch
        x = x.view(x.size(0), self.input_ts, -1).float()
        y = y.view(y.size(0), -1).float()
        return x, y


class PermuteSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)
