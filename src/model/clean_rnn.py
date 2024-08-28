import pdb
import torch
import torch.nn as nn
import lightning as L
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

# basic block


class PermuteSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.8, batch_norm=True):
        super(LinearBlock, self).__init__()
        half_dim = input_dim // 2
        self.linear = nn.Linear(input_dim, half_dim)
        self.linear2 = nn.Linear(half_dim, output_dim)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(half_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm:
            x = x.permute(0, 2, 1)
            x = self.batch_norm(x)
            x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# RNN model


class LSTMSeq(nn.Module):
    def __init__(self, input_dim, in_seq_len, out_seq_len, hidden_dim=32, dropout=0.8, use_prob=False):
        super().__init__()
        self.input_dim = input_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_prob = use_prob

        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=hidden_dim,
                               num_layers=2,
                               dropout=dropout,
                               batch_first=True)

        self.heat_decoder = nn.LSTM(input_size=hidden_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=2,
                                    dropout=dropout,
                                    batch_first=True)
        self.cool_decoder = nn.LSTM(input_size=hidden_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=2,
                                    dropout=dropout,
                                    batch_first=True)
        self.heat_decoder_linear = LinearBlock(hidden_dim, 1, dropout=dropout)
        self.cool_decoder_linear = LinearBlock(hidden_dim, 1, dropout=dropout)
        if use_prob:
            self.heat_tigger = LinearBlock(hidden_dim, 1, dropout=dropout)
            self.cool_trigger = LinearBlock(hidden_dim, 1, dropout=dropout)

    def forward(self, x):
        z, encode_hidden = self.encoder(x)
        heat_latent, _ = self.heat_decoder(z, encode_hidden)
        cool_latent, _ = self.cool_decoder(z, encode_hidden)
        heat_hat = self.heat_decoder_linear(heat_latent)
        cool_hat = self.cool_decoder_linear(cool_latent)
        if self.use_prob:
            heat_prob = self.heat_tigger(heat_latent)
            cool_prob = self.cool_trigger(cool_latent)
            return heat_hat, cool_hat, heat_prob, cool_prob
        return heat_hat, cool_hat

# Transformer model


class TransNet(nn.Module):
    def __init__(self, input_dim, in_seq_len, out_seq_len, hidden_dim=32, dropout=0.8, use_prob=False):
        super().__init__()
        self.input_dim = input_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_prob = use_prob

        self.num_encoder_layers = 2
        self.num_decoder_layers = 2

        self.linear_embed = nn.Sequential(
            nn.Linear(26, 256), PermuteSeq(), nn.BatchNorm1d(256), PermuteSeq())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=256, batch_first=True)
        encoder_norm = nn.Sequential(nn.LayerNorm(256), PermuteSeq(
        ), nn.InstanceNorm1d(256, affine=True), PermuteSeq())

        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, batch_first=True)
        self.heat_decoder = nn.TransformerEncoder(
            decoder_layer, self.num_decoder_layers, encoder_norm)
        self.cool_decoder = nn.TransformerEncoder(
            decoder_layer, self.num_decoder_layers, encoder_norm)

        self.heat_decoder_linear = LinearBlock(hidden_dim, 1, dropout=dropout)
        self.cool_decoder_linear = LinearBlock(hidden_dim, 1, dropout=dropout)
        if use_prob:
            self.heat_tigger = LinearBlock(hidden_dim, 1, dropout=dropout)
            self.cool_trigger = LinearBlock(hidden_dim, 1, dropout=dropout)

        self.save_hyperparameters()

    def forward(self, x):
        embed_x = self.linear_embed(x)
        z = self.encoder(embed_x)
        heat_latent = self.heat_decoder(z)
        cool_latent = self.cool_decoder(z)
        heat_hat = self.heat_decoder_linear(heat_latent)
        cool_hat = self.cool_decoder_linear(cool_latent)
        if self.use_prob:
            heat_prob = self.heat_tigger(heat_latent)
            cool_prob = self.cool_trigger(cool_latent)
            return heat_hat, cool_hat, heat_prob, cool_prob
        return heat_hat, cool_hat


def deterministic_loss(heat_hat, cool_hat, y):
    heat_loss = nn.functional.mse_loss(heat_hat, y[:, :, 0].unsqueeze(2))
    cool_loss = nn.functional.mse_loss(cool_hat, y[:, :, 1].unsqueeze(2))
    return heat_loss, cool_loss


def probablistic_loss(heat_hat, cool_hat, heat_prob, cool_prob, y, h_zero, c_zero):
    h_mask = (y[:, :, 0] != h_zero)
    c_mask = (y[:, :, 1] != c_zero)
    heat_prob_loss = nn.functional.binary_cross_entropy_with_logits(
        heat_prob, h_mask.unsqueeze(2).float())
    cool_prob_loss = nn.functional.binary_cross_entropy_with_logits(
        cool_prob, c_mask.unsqueeze(2).float())
    heat_loss = nn.functional.mse_loss(heat_hat, y[:, :, 0].unsqueeze(2))
    cool_loss = nn.functional.mse_loss(cool_hat, y[:, :, 1].unsqueeze(2))
    return heat_loss, cool_loss, heat_prob_loss, cool_prob_loss


class CityModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

        self.in_seq_len = config.in_seq_len
        self.out_seq_len = config.out_seq_len

        # normalization
        self.scaling = config.scaling
        self.h_zero = torch.tensor(-self.scaling.H_MEAN /
                                   self.scaling.H_STD).to(torch.float16)
        self.c_zero = torch.tensor(-self.scaling.C_MEAN /
                                   self.scaling.C_STD).to(torch.float16)

        # learning rate
        self.learning_rate = config.learning_rate
        self.lr_warmup_epochs = config.lr_warmup_epochs
        self.num_batches = config.num_batches
        self.num_epochs = config.num_epochs

        # deterministic or probablistic
        if config.loss == 'deterministic':
            self.criterion = deterministic_loss
            self.use_prob = False
        else:
            self.criterion = probablistic_loss
            self.use_prob = True

        # load model
        self.model = LSTMSeq(input_dim=self.input_dim, in_seq_len=self.in_seq_len,
                             out_seq_len=self.out_seq_len, hidden_dim=self.hidden_dim, dropout=config.dropout, use_prob=self.use_prob)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def split_reshape(self, batch):
        x, y = batch
        x = x.view(x.size(0), self.in_seq_len, -1).float()
        y = y.view(y.size(0), self.out_seq_len, -1).float()
        return x, y

    def inverse_transform_load(self, heat, cool):
        return heat * self.scaling.H_STD + self.scaling.H_MEAN, cool * self.scaling.C_STD + self.scaling.C_MEAN

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate)
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=(self.num_batches * self.lr_warmup_epochs),
            num_training_steps=(self.num_batches * self.num_epochs),
            num_cycles=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }

    def general_process(self, batch, mode):
        x, y = self.split_reshape(batch)

        if not self.use_prob:
            heat_hat, cool_hat = self(x)
            heat_loss, cool_loss = self.criterion(heat_hat, cool_hat, y)
            loss = heat_loss + cool_loss
            self.log_dict({f"{mode}/heat_loss": heat_loss, f"{mode}/cool_loss": cool_loss,
                           f"{mode}/total_loss": loss}, prog_bar=True)
        else:
            heat_hat, cool_hat, heat_prob, cool_prob = self(x)
            heat_loss, cool_loss, heat_prob_loss, cool_prob_loss = self.criterion(
                heat_hat, cool_hat, heat_prob, cool_prob, y, self.h_zero, self.c_zero)
            loss = heat_loss + cool_loss + heat_prob_loss + cool_prob_loss
            self.log_dict({f"{mode}/heat_loss": heat_loss, f"{mode}/cool_loss": cool_loss,
                           f"{mode}/heat_prob_loss": heat_prob_loss, f"{mode}/cool_prob_loss": cool_prob_loss,
                           f"{mode}/total_loss": loss}, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.general_process(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_process(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.general_process(batch, "test")
