from typing import Any
import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, StepLR
from model.base import RNNNet, PermuteSeq
from configs.configuration import QUANTILES
from utils.criterions import QuantileLoss
import pdb


class LSTMSeq(L.LightningModule):
    def __init__(self, input_dim: int, input_seq_len: int, output_seq_len: int,
                 encode_layers: int, decode_layers: int,
                 hidden_dim: int, dropout: float, scaling) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.encode_layers = encode_layers
        self.decode_layers = decode_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.scaling = scaling
        self.save_hyperparameters(ignore=['scaling'])

        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.encode_layers,
                               dropout=self.dropout,
                               batch_first=True)
        self.heat_lstm_decoder = nn.LSTM(input_size=self.hidden_dim,
                                         hidden_size=self.hidden_dim,
                                         num_layers=self.decode_layers,
                                         dropout=self.dropout,
                                         batch_first=True)
        self.cool_lstm_decoder = nn.LSTM(input_size=self.hidden_dim,
                                         hidden_size=self.hidden_dim,
                                         num_layers=self.decode_layers,
                                         dropout=self.dropout,
                                         batch_first=True)
        self.heat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            PermuteSeq(),
            nn.BatchNorm1d(hidden_dim//2),
            PermuteSeq(),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            PermuteSeq(),
            nn.BatchNorm1d(hidden_dim//2),
            PermuteSeq(),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1))
    
    def forward(self, x):
        z, encode_hidden = self.encoder(x)
        heat_hat_seq, _ = self.heat_lstm_decoder(z, encode_hidden)
        cool_hat_seq, _ = self.cool_lstm_decoder(z, encode_hidden)
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

    def predict(self, x):
        heat_hat, cool_hat = self.forward(x)
        heat_hat = heat_hat * self.scaling.H_STD + self.scaling.H_MEAN
        cool_hat = cool_hat * self.scaling.C_STD + self.scaling.C_MEAN
        return heat_hat, cool_hat

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        x, y = self.split_reshape(batch)
        # inverse transform the load
        y[:, :, 0] = y[:, :, 0] * self.h_std + self.h_mean
        y[:, :, 1] = y[:, :, 1] * self.c_std + self.c_mean
        return self.predict(x), y


class RNNSeqNet(RNNNet):
    def __init__(self, input_dim, input_ts, output_ts, hidden_dim, dropout, scaling):
        super().__init__(input_dim, input_ts, hidden_dim)
        self.output_ts = output_ts
        self.scaling = scaling

        self.encoder = nn.LSTM(input_size=self.input_dim,
                               hidden_size=hidden_dim,
                               num_layers=2,
                               dropout=dropout,
                               batch_first=True)
        self.heat_decoder_seq = nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        num_layers=2,
                                        dropout=dropout,
                                        batch_first=True)
        self.cool_decoder_seq = nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        num_layers=2,
                                        dropout=dropout,
                                        batch_first=True)
        self.heat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), PermuteSeq(), nn.BatchNorm1d(hidden_dim//2), PermuteSeq(), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), PermuteSeq(), nn.BatchNorm1d(hidden_dim//2), PermuteSeq(), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

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

    def predict(self, x):
        heat_hat, cool_hat = self.forward(x)
        heat_hat = heat_hat * self.scaling.H_STD + self.scaling.H_MEAN
        cool_hat = cool_hat * self.scaling.C_STD + self.scaling.C_MEAN
        return heat_hat, cool_hat

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        x, y = self.split_reshape(batch)
        # inverse transform the load
        y[:, :, 0] = y[:, :, 0] * self.h_std + self.h_mean
        y[:, :, 1] = y[:, :, 1] * self.c_std + self.c_mean
        return self.predict(x), y


class RNNEmbedNet(RNNSeqNet):
    def __init__(self, input_dim, input_ts, output_ts):
        super().__init__(input_dim, input_ts, output_ts)
        self.linear_embed = nn.Linear(25, 256)
        self.encoder = nn.LSTM(input_size=256,
                               hidden_size=256,
                               num_layers=2,
                               batch_first=True)
        self.save_hyperparameters()

    def forward(self, x):
        return super().forward(self.linear_embed(x))


class RNNSeqNetV2(RNNSeqNet):
    def __init__(self, input_dim, input_ts, output_ts, hidden_dim, dropout,
                 scaling):
        super().__init__(input_dim, input_ts, output_ts,
                         hidden_dim, dropout, scaling)
        self.scaling = scaling
        self.h_zero = torch.tensor(-scaling.H_MEAN /
                                   scaling.H_STD).to(torch.float16)
        self.c_zero = torch.tensor(-scaling.C_MEAN /
                                   scaling.C_STD).to(torch.float16)
        self.h_trigger_ratio = scaling.H_TRIGGER_RATIO
        self.c_trigger_ratio = scaling.C_TRIGGER_RATIO
        self.quantiles = QUANTILES
        self.quantile_loss = QuantileLoss(self.quantiles)
        self.sigmoid = nn.Sigmoid()

        self.validation_confusion_matrix = {'heat': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                                            'cool': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}
        half_dim = hidden_dim // 2
        self.heat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, len(self.quantiles)))
        self.cool_decoder = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, len(self.quantiles)))
        self.heat_tigger = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, 1))
        self.cool_trigger = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, 1))
        self.save_hyperparameters(ignore=['scaling'])

    def forward(self, x):
        z, encode_hidden = self.encoder(x)
        heat_hat_seq, _ = self.heat_decoder_seq(z, encode_hidden)
        cool_hat_seq, _ = self.cool_decoder_seq(z, encode_hidden)
        heat_hat = self.heat_decoder(heat_hat_seq)
        cool_hat = self.cool_decoder(cool_hat_seq)
        heat_prob = self.heat_tigger(heat_hat_seq)
        cool_prob = self.cool_trigger(cool_hat_seq)
        return heat_hat, cool_hat, heat_prob, cool_prob

    def criterion(self, heat_hat, cool_hat, heat_prob, cool_prob, y, threshold=0.5):
        h_mask = (y[:, :, 0] != self.h_zero)
        c_mask = (y[:, :, 1] != self.c_zero)
        heat_prob_loss = nn.functional.binary_cross_entropy_with_logits(
            heat_prob, h_mask.unsqueeze(2).float())
        cool_prob_loss = nn.functional.binary_cross_entropy_with_logits(
            cool_prob, c_mask.unsqueeze(2).float())
        heat_loss = self.quantile_loss(
            heat_hat[h_mask], y[:, :, 0].unsqueeze(2)[h_mask]).sum()
        cool_loss = self.quantile_loss(
            cool_hat[c_mask], y[:, :, 1].unsqueeze(2)[c_mask]).sum()
        return heat_loss, cool_loss, heat_prob_loss, cool_prob_loss

    def training_step(self, batch, batch_idx):
        x, y = self.split_reshape(batch)

        heat_hat, cool_hat, heat_prob, cool_prob = self(x)

        heat_loss, cool_loss, heat_prob_loss, cool_prob_loss = self.criterion(
            heat_hat, cool_hat, heat_prob, cool_prob, y)
        loss = heat_loss + cool_loss + heat_prob_loss + cool_prob_loss
        self.log_cls_result(y, self.sigmoid(heat_prob),
                            self.sigmoid(cool_prob))

        self.log_dict({"train/heat_loss": heat_loss, "train/cool_loss": cool_loss,
                       "train/heat_prob_loss": heat_prob_loss, "train/cool_prob_loss": cool_prob_loss,
                       "train/total_loss": loss, "global_step": self.global_step})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.split_reshape(batch)

        heat_hat, cool_hat, heat_prob, cool_prob = self(x)

        heat_loss, cool_loss, heat_prob_loss, cool_prob_loss = self.criterion(
            heat_hat, cool_hat, heat_prob, cool_prob, y)

        heat_loss = 0 if torch.isnan(heat_loss) else heat_loss
        cool_loss = 0 if torch.isnan(cool_loss) else cool_loss
        loss = heat_prob_loss + cool_prob_loss + heat_loss + cool_loss
        self.record_cls_output(y, self.sigmoid(
            heat_prob), self.sigmoid(cool_prob))

        self.log_dict({"val/heat_loss": heat_loss, "val/cool_loss": cool_loss,
                       "val/heat_prob_loss": heat_prob_loss, "val/cool_prob_loss": cool_prob_loss,
                       "val/total_loss": loss, "global_step": self.global_step})
        return loss

    def on_validation_epoch_end(self):
        for h_or_c in ('heat', 'cool'):
            # calculate accuracy, precision, recall, and f1 score
            sin_confu_mat = self.validation_confusion_matrix[h_or_c]
            accuracy = (sin_confu_mat['tp'] + sin_confu_mat['tn']) / (
                sin_confu_mat['tp'] + sin_confu_mat['tn'] + sin_confu_mat['fp'] + sin_confu_mat['fn'])
            precision = sin_confu_mat['tp'] / \
                (sin_confu_mat['tp'] + sin_confu_mat['fp'])
            recall = sin_confu_mat['tp'] / \
                (sin_confu_mat['tp'] + sin_confu_mat['fn'])
            f1 = 2 * precision * recall / (precision + recall)
            # log the results
            self.log_dict({f"val_prob/{h_or_c}/accuracy": accuracy, f"val_prob/{h_or_c}/precision": precision,
                           f"val_prob/{h_or_c}/recall": recall, f"val_prob/{h_or_c}/f1": f1})
        self.validation_confusion_matrix = {'heat': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                                            'cool': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}

    def test_step(self, batch, batch_idx):
        x, y = self.split_reshape(batch)

        heat_hat, cool_hat, heat_prob, cool_prob = self(x)

        heat_loss, cool_loss, heat_prob_loss, cool_prob_loss = self.criterion(
            heat_hat, cool_hat, heat_prob, cool_prob, y)

        heat_loss = 0 if torch.isnan(heat_loss) else heat_loss
        cool_loss = 0 if torch.isnan(cool_loss) else cool_loss
        loss = heat_prob_loss + cool_prob_loss + heat_loss + cool_loss

        self.log_dict({"test/heat_loss": heat_loss, "test/cool_loss": cool_loss,
                       "test/heat_prob_loss": heat_prob_loss, "test/cool_prob_loss": cool_prob_loss,
                       "test/total_loss": loss, "global_step": self.global_step})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler1 = LinearLR(
            optimizer, start_factor=0.01, total_iters=50)
        scheduler3 = CosineAnnealingLR(optimizer, T_max=100)
        scheduler = SequentialLR(optimizer, schedulers=[
                                 scheduler1, scheduler3], milestones=[300])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }

    def predict(self, x):
        heat_hat, cool_hat, heat_prob, cool_prob = self(x)
        heat_hat = heat_hat * self.scaling.H_STD + self.scaling.H_MEAN
        cool_hat = cool_hat * self.scaling.C_STD + self.scaling.C_MEAN
        idx = self.quantiles.index(0.5)
        heat_hat = heat_hat[:, :, idx]
        cool_hat = cool_hat[:, :, idx]
        heat_hat[(heat_prob < self.h_trigger_ratio).squeeze()] = 0
        cool_hat[(cool_prob < self.c_trigger_ratio).squeeze()] = 0
        return heat_hat, cool_hat

    def record_cls_output(self, y, heat_prob, cool_prob):
        h_trigger = (y[..., 0] != self.h_zero)
        c_trigger = (y[..., 1] != self.c_zero)
        h_tri_pred = (heat_prob > self.h_trigger_ratio).squeeze()
        c_tri_pred = (cool_prob > self.c_trigger_ratio).squeeze()
        # calculate confusion matrix
        for y_true, y_pred, h_or_c in zip((h_trigger, c_trigger), (h_tri_pred, c_tri_pred), ('heat', 'cool')):
            tp = (y_true & y_pred).sum()
            tn = (~y_true & ~y_pred).sum()
            fp = (~y_true & y_pred).sum()
            fn = (y_true & ~y_pred).sum()
            self.validation_confusion_matrix[h_or_c]['tp'] += tp
            self.validation_confusion_matrix[h_or_c]['tn'] += tn
            self.validation_confusion_matrix[h_or_c]['fp'] += fp
            self.validation_confusion_matrix[h_or_c]['fn'] += fn
        return

    def log_cls_result(self, y, heat_prob, cool_prob, stage='train'):
        h_trigger = (y[..., 0] != self.h_zero)
        c_trigger = (y[..., 1] != self.c_zero)
        h_tri_pred = (heat_prob > self.h_trigger_ratio).squeeze()
        c_tri_pred = (cool_prob > self.c_trigger_ratio).squeeze()
        # calculate confusion matrix
        for y_true, y_pred, h_or_c in zip((h_trigger, c_trigger), (h_tri_pred, c_tri_pred), ('heat', 'cool')):
            tp = (y_true & y_pred).sum()
            tn = (~y_true & ~y_pred).sum()
            fp = (~y_true & y_pred).sum()
            fn = (y_true & ~y_pred).sum()
            # calculate accuracy, precision, recall, and f1 score
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
            # log the results
            self.log_dict({f"{stage}_prob/{h_or_c}/accuracy": accuracy, f"{stage}_prob/{h_or_c}/precision": precision,
                           f"{stage}_prob/{h_or_c}/recall": recall, f"{stage}_prob/{h_or_c}/f1": f1})

    def log_load_difference(self, y, heat_hat, cool_hat):
        h_mask = (y[:, :, 0].unsqueeze(2) != self.h_zero)
        c_mask = (y[:, :, 1].unsqueeze(2) != self.c_zero)
        h_diff = torch.abs(y[:, :, 0].unsqueeze(
            2)[h_mask] - heat_hat[h_mask]).mean()
        c_diff = torch.abs(y[:, :, 1].unsqueeze(
            2)[c_mask] - cool_hat[c_mask]).mean()
        self.log_dict({"train/heat_diff": h_diff, "train/cool_diff": c_diff})

    def inverse_transform_load(self, y):
        y[:, :, 0] = y[:, :, 0] * self.scaling.H_STD + self.scaling.H_MEAN
        y[:, :, 1] = y[:, :, 1] * self.scaling.C_STD + self.scaling.C_MEAN
        return y
