from model.rnn import RNNSeqNet, RNNSeqNetV2, LSTMSeq, LSTMSeqQuantile, LSTMSeqProb
from configs.configuration import QUANTILES
from utils.criterions import QuantileLoss
from torchvision.ops import sigmoid_focal_loss
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


class TransSeq(LSTMSeq):
    def __init__(self, input_dim: int, input_seq_len: int, output_seq_len: int,
                 encode_layers: int, decode_layers: int,
                 hidden_dim: int, dropout: float, scaling) -> None:
        super().__init__(input_dim, input_seq_len, output_seq_len,
                         encode_layers, decode_layers, hidden_dim, dropout, scaling)

        self.linear_embed = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True)
        encoder_norm = nn.Sequential(nn.LayerNorm(hidden_dim), PermuteSeq(
        ), nn.InstanceNorm1d(hidden_dim, affine=True), PermuteSeq())

        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.encode_layers, encoder_norm)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True)
        self.heat_decoder_seq = nn.TransformerEncoder(
            decoder_layer, self.decode_layers, encoder_norm)
        self.cool_decoder_seq = nn.TransformerEncoder(
            decoder_layer, self.decode_layers, encoder_norm)

        self.save_hyperparameters(ignore=['scaling'])

    def split_reshape(self, batch):
        x, y = batch
        x = x.view(x.size(0), self.input_seq_len, -1).float()
        y = y.view(y.size(0), self.output_seq_len, -1).float()
        return x, y

    def forward(self, x):
        embed_x = self.linear_embed(x)
        z = self.encoder(embed_x)
        heat_hat_seq = self.heat_decoder_seq(z)
        cool_hat_seq = self.cool_decoder_seq(z)
        heat_hat = self.heat_decoder(heat_hat_seq)
        cool_hat = self.cool_decoder(cool_hat_seq)
        return heat_hat, cool_hat


class TransSeqQuantile(TransSeq):
    def __init__(self, input_dim: int, input_seq_len: int, output_seq_len: int,
                 encode_layers: int, decode_layers: int,
                 hidden_dim: int, dropout: float, scaling) -> None:
        super().__init__(input_dim, input_seq_len, output_seq_len,
                         encode_layers, decode_layers, hidden_dim, dropout, scaling)
        self.quantiles = QUANTILES
        self.quantile_loss = QuantileLoss(self.quantiles)
        self.sigmoid = nn.Sigmoid()

        half_dim = hidden_dim // 2
        self.heat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, len(self.quantiles)))
        self.cool_decoder = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, len(self.quantiles)))
        self.save_hyperparameters(ignore=['scaling'])

    def criterion(self, heat_hat, cool_hat, y):
        heat_loss = self.quantile_loss(heat_hat, y[:, :, 0].unsqueeze(2)).sum()
        cool_loss = self.quantile_loss(cool_hat, y[:, :, 1].unsqueeze(2)).sum()
        return heat_loss, cool_loss

    def log_load_difference(self, y, heat_hat, cool_hat, stage='train'):
        heat_true, cool_true = self.inverse_transform_load(
            y[:, :, 0], y[:, :, 1])
        idx = self.quantiles.index(0.5)
        heat_hat, cool_hat = self.inverse_transform_load(
            heat_hat[:, :, idx], cool_hat[:, :, idx])
        heat_diff = nn.functional.l1_loss(heat_true, heat_hat)
        cool_diff = nn.functional.l1_loss(cool_true, cool_hat)
        self.log_dict({f'{stage}/heat_diff': heat_diff, f'{stage}/cool_diff': cool_diff,
                      f'{stage}/total_diff': heat_diff + cool_diff})

    def generate_load(self, heat_hat, cool_hat):
        pred_heat, pred_cool = self.inverse_transform_load(heat_hat, cool_hat)
        idx = self.quantiles.index(0.5)
        pred_heat = pred_heat[:, :, idx]
        pred_cool = pred_cool[:, :, idx]
        return pred_heat, pred_cool


class TransSeqProb(TransSeqQuantile):
    def __init__(self, input_dim: int, input_seq_len: int, output_seq_len: int,
                 encode_layers: int, decode_layers: int, hidden_dim: int, dropout: float,
                 scaling) -> None:
        super().__init__(input_dim, input_seq_len, output_seq_len,
                         encode_layers, decode_layers, hidden_dim, dropout, scaling)

        self.h_zero = -scaling.H_MEAN / scaling.H_STD
        self.c_zero = -scaling.C_MEAN / scaling.C_STD
        self.h_trigger_ratio = scaling.H_TRIGGER_RATIO
        self.c_trigger_ratio = scaling.C_TRIGGER_RATIO

        self.validation_confusion_matrix = {'heat': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                                            'cool': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}

        half_dim = hidden_dim // 2

        self.heat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, 1))
        self.heat_tigger = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, 1))
        self.cool_trigger = nn.Sequential(
            nn.Linear(hidden_dim, half_dim), PermuteSeq(), nn.BatchNorm1d(half_dim), PermuteSeq(), nn.ReLU(), nn.Linear(half_dim, 1))
        self.save_hyperparameters(ignore=['scaling'])

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

    def generate_load(self, heat_hat, cool_hat, heat_prob, cool_prob):
        heat_hat, cool_hat = self.inverse_transform_load(heat_hat, cool_hat)
        heat_hat[(torch.sigmoid(heat_prob) < self.h_trigger_ratio).squeeze()] = 0
        cool_hat[(torch.sigmoid(cool_prob) < self.c_trigger_ratio).squeeze()] = 0
        return heat_hat, cool_hat

    def log_load_difference(self, y, heat_hat, cool_hat, heat_prob, cool_prob, stage='train'):
        heat_true, cool_true = self.inverse_transform_load(
            y[:, :, 0], y[:, :, 1])
        heat_hat, cool_hat = self.generate_load(
            heat_hat, cool_hat, heat_prob, cool_prob)
        heat_diff = nn.functional.l1_loss(heat_true.unsqueeze(2), heat_hat)
        cool_diff = nn.functional.l1_loss(cool_true.unsqueeze(2), cool_hat)
        self.log_dict({f'{stage}/heat_diff': heat_diff, f'{stage}/cool_diff': cool_diff,
                      f'{stage}/total_diff': heat_diff + cool_diff})

    def criterion(self, heat_hat, cool_hat, heat_prob, cool_prob, y, threshold=0.5):
        h_mask = (y[:, :, 0] != self.h_zero)
        c_mask = (y[:, :, 1] != self.c_zero)
        # heat_prob_loss = nn.functional.binary_cross_entropy_with_logits(
        #     heat_prob, h_mask.unsqueeze(2).float())
        # cool_prob_loss = nn.functional.binary_cross_entropy_with_logits(
        #     cool_prob, c_mask.unsqueeze(2).float())
        heat_prob_loss = sigmoid_focal_loss(
            heat_prob, h_mask.unsqueeze(2).float(), reduction='mean')
        cool_prob_loss = nn.functional.binary_cross_entropy_with_logits(
            cool_prob, c_mask.unsqueeze(2).float(), reduction='mean')
        heat_loss = nn.functional.mse_loss(
            heat_hat, y[:, :, 0].unsqueeze(2))
        cool_loss = nn.functional.mse_loss(
            cool_hat, y[:, :, 1].unsqueeze(2))
        return heat_loss, cool_loss, heat_prob_loss, cool_prob_loss

    def training_step(self, batch, batch_idx):
        x, y = self.split_reshape(batch)

        heat_hat, cool_hat, heat_prob, cool_prob = self(x)

        heat_loss, cool_loss, heat_prob_loss, cool_prob_loss = self.criterion(
            heat_hat, cool_hat, heat_prob, cool_prob, y)
        loss = heat_loss + cool_loss + heat_prob_loss + cool_prob_loss
        self.log_dict({"train/heat_loss": heat_loss, "train/cool_loss": cool_loss,
                       "train/heat_prob_loss": heat_prob_loss, "train/cool_prob_loss": cool_prob_loss,
                       "train/total_loss": loss, "global_step": self.global_step})
        # log actual load diff
        self.log_load_difference(
            y, heat_hat, cool_hat, heat_prob, cool_prob, stage='train')
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
        # log actual load diff
        self.log_load_difference(
            y, heat_hat, cool_hat, heat_prob, cool_prob, stage='val')
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
        # log actual load diff
        self.log_load_difference(
            y, heat_hat, cool_hat, heat_prob, cool_prob, stage='test')
        return loss

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
