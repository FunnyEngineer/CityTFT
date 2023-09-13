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
    def __init__(self, input_dim, input_ts):
        super().__init__()
        self.input_dim = input_dim
        self.input_ts = input_ts

        self.encoder = nn.LSTM(input_size=input_dim,
                               hidden_size=256,
                               num_layers=2,
                               batch_first=True)
        self.heat_decoder = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1))

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


class RNNSeqNet(RNNNet):
    def __init__(self, input_dim, input_ts, output_ts):
        super().__init__(input_dim, input_ts)
        self.output_ts = output_ts

        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=256,
                               num_layers=2,
                               batch_first=True)
        self.heat_decoder_seq = nn.LSTM(input_size=256,
                                        hidden_size=256,
                                        num_layers=2,
                                        batch_first=True)
        self.cool_decoder_seq = nn.LSTM(input_size=256,
                                        hidden_size=256,
                                        num_layers=2,
                                        batch_first=True)
        self.heat_decoder = nn.Sequential(
            nn.Linear(256, 128), PermuteSeq(), nn.BatchNorm1d(128), PermuteSeq(), nn.ReLU(), nn.Linear(128, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(256, 128), PermuteSeq(), nn.BatchNorm1d(128), PermuteSeq(), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        z, encode_hidden = self.encoder(x)
        heat_hat_seq, _ = self.heat_decoder_seq(z, encode_hidden)
        cool_hat_seq, _ = self.cool_decoder_seq(z, encode_hidden)
        heat_hat = self.heat_decoder(heat_hat_seq)
        cool_hat = self.cool_decoder(cool_hat_seq)
        return heat_hat, cool_hat

    def split_reshape(self, batch):
        x, y = batch
        x = x.view(x.size(0), self.input_ts, -1).float()
        y = y.view(y.size(0), self.output_ts, -1).float()
        return x, y

    def criterion(self, heat_hat, cool_hat, y):
        heat_loss = nn.functional.mse_loss(heat_hat, y[:, :, 0].unsqueeze(2))
        cool_loss = nn.functional.mse_loss(cool_hat, y[:, :, 1].unsqueeze(2))
        return heat_loss, cool_loss


class TransformerSeqNet(RNNSeqNet):
    def __init__(self, input_dim, input_ts, output_ts):
        super().__init__(input_dim, input_ts, output_ts)
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2

        self.linear_embed = nn.Linear(25, 256)
        self.linear_embed_ts = nn.Linear(1, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=256, batch_first=True)
        encoder_norm = nn.LayerNorm(256)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=256, batch_first=True)
        decoder_norm = nn.LayerNorm(256)
        self.heat_decoder_seq = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=self.num_decoder_layers, norm=decoder_norm)
        self.cool_decoder_seq = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=self.num_decoder_layers, norm=decoder_norm)

        self.save_hyperparameters()

    def split_reshape(self, batch):
        x, y, ts = batch
        x = x.view(x.size(0), self.input_ts, -1).float()
        y = y.view(y.size(0), self.output_ts, -1).float()
        ts = ts.view(ts.size(0), self.output_ts, -1).float()
        x = torch.cat((x, ts), dim=-1)
        return x, y

    def forward(self, x):
        ts = x[:, :, -1].unsqueeze(2)
        x = x[:, :, :-1]

        embed_x = self.linear_embed(x)
        z = self.encoder(embed_x)
        heat_hat_seq = self.heat_decoder_seq(self.linear_embed_ts(ts), z)
        cool_hat_seq = self.cool_decoder_seq(self.linear_embed_ts(ts), z)
        heat_hat = self.heat_decoder(heat_hat_seq)
        cool_hat = self.cool_decoder(cool_hat_seq)
        return heat_hat, cool_hat
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
    

class HybridRNNAttenNet(RNNSeqNet):
    def __init__(self, input_dim, input_ts, output_ts):
        super().__init__(input_dim, input_ts, output_ts)
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2

        self.linear_embed = nn.Linear(25, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=256, batch_first=True)
        encoder_norm = nn.LayerNorm(256)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm)
        # self.encoder = nn.Transformer(d_model=256, num_decoder_layers=2, num_encoder_layers=2, dim_feedforward=256, batch_first=True)


    def forward(self, x):
        embed_x = self.linear_embed(x)
        z = self.encoder(embed_x)
        pdb.set_trace()
        heat_hat_seq, _ = self.heat_decoder_seq(z)
        cool_hat_seq, _ = self.cool_decoder_seq(z)
        heat_hat = self.heat_decoder(heat_hat_seq)
        cool_hat = self.cool_decoder(cool_hat_seq)
        return heat_hat, cool_hat