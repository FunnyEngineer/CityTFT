from model.rnn import RNNSeqNet, RNNSeqNetV2
from model.base import PermuteSeq
import torch
import torch.nn as nn
import pdb

torch.set_float32_matmul_precision('high')

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


class TransNetV2(RNNSeqNetV2):
    def __init__(self, input_dim, input_ts, output_ts):
        super().__init__(input_dim, input_ts, output_ts)
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2

        self.linear_embed = nn.Sequential(nn.Linear(26, 256), PermuteSeq(), nn.BatchNorm1d(256), PermuteSeq())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=256, batch_first=True)
        encoder_norm = nn.Sequential(nn.LayerNorm(256), PermuteSeq(), nn.InstanceNorm1d(256, affine=True), PermuteSeq())
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, batch_first=True)
        self.heat_decoder_seq = nn.TransformerEncoder(
            decoder_layer, self.num_decoder_layers, encoder_norm)
        self.cool_decoder_seq = nn.TransformerEncoder(
            decoder_layer, self.num_decoder_layers, encoder_norm)
        self.heat_tigger = nn.Sequential(
            nn.Linear(256, 128), PermuteSeq(), nn.BatchNorm1d(128), PermuteSeq(), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())
        self.cool_trigger = nn.Sequential(
            nn.Linear(256, 128), PermuteSeq(), nn.BatchNorm1d(128), PermuteSeq(), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

        self.save_hyperparameters()

    def split_reshape(self, batch):
        x, y = batch
        x = x.view(x.size(0), self.input_ts, -1).float()
        y = y.view(y.size(0), self.output_ts, -1).float()
        return x, y

    def forward(self, x):

        embed_x = self.linear_embed(x)
        z = self.encoder(embed_x)
        heat_hat_seq = self.heat_decoder_seq(z)
        cool_hat_seq = self.cool_decoder_seq(z)
        heat_hat = self.heat_decoder(heat_hat_seq)
        cool_hat = self.cool_decoder(cool_hat_seq)
        heat_prob = self.heat_tigger(heat_hat_seq)
        cool_prob = self.cool_trigger(cool_hat_seq)
        return heat_hat, cool_hat, heat_prob, cool_prob


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
        heat_hat_seq, _ = self.heat_decoder_seq(z)
        cool_hat_seq, _ = self.cool_decoder_seq(z)
        heat_hat = self.heat_decoder(heat_hat_seq)
        cool_hat = self.cool_decoder(cool_hat_seq)
        return heat_hat, cool_hat
