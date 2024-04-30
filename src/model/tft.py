from utils.criterions import QuantileLoss
import lightning as L
from torch.nn import LayerNorm
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import UninitializedParameter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, StepLR
from typing import Dict, Tuple, Optional, List

from model.base import PermuteSeq

MAKE_CONVERT_COMPATIBLE = os.environ.get("TFT_SCRIPTING", None) is not None


class LinearBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = LayerNorm(out_channels)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = PermuteSeq()(x)
        x = self.bn1(x)
        x = PermuteSeq()(x)
        x = self.act1(x)
        return x


class UninterpretableTFT(L.LightningModule):
    def __init__(
            self,
            seq_len: int,
            static_dim: int,
            temporal_dim: int,
            hidden_dim: int,
            scaling,
            encode_layers: int = 2,
            decode_layers: int = 2,
    ) -> None:
        super().__init__()
        self.scaling = scaling
        self.h_zero = -scaling.H_MEAN / scaling.H_STD
        self.c_zero = -scaling.C_MEAN / scaling.C_STD
        self.h_trigger_ratio = scaling.H_TRIGGER_RATIO
        self.c_trigger_ratio = scaling.C_TRIGGER_RATIO
        self.validation_confusion_matrix = {'heat': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                                            'cool': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}


        self.static_encoder = LinearBasicBlock(static_dim, hidden_dim)

        self.temporal_encoder = LinearBasicBlock(temporal_dim, hidden_dim)

        self.encoder = nn.LSTM(
            input_size=hidden_dim*2,
            hidden_size=hidden_dim*2,
            num_layers=encode_layers,
            batch_first=True)

        self.static_enricher = LinearBasicBlock(hidden_dim * 3, hidden_dim * 3)

        self.heat_decoder_seq = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim * 3, nhead=8, batch_first=True), 
            decode_layers, 
            norm=LayerNorm(hidden_dim * 3))

        self.cool_decoder_seq = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim * 3, nhead=8, batch_first=True), 
            decode_layers, 
            norm=LayerNorm(hidden_dim * 3))
        
        self.heat_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), PermuteSeq(), nn.BatchNorm1d(hidden_dim), PermuteSeq(), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.cool_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), PermuteSeq(), nn.BatchNorm1d(hidden_dim), PermuteSeq(), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.heat_tigger = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), PermuteSeq(), nn.BatchNorm1d(hidden_dim), PermuteSeq(), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.cool_trigger = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), PermuteSeq(), nn.BatchNorm1d(hidden_dim), PermuteSeq(), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.save_hyperparameters(ignore=['scaling'])

    def forward(self, x: Tensor) -> Tensor:
        sta_x, tem_x = x
        seq_len = tem_x.size(1)
        # encoder
        # 1. static data -> embedded static
        sta_x = self.static_encoder(sta_x).repeat(1, seq_len, 1)
        # 2. temporal data concat with embedded static
        tem_x = self.temporal_encoder(tem_x)
        # 3. embeded static + temporal data
        # 4. LSTM encoder -> encoded temporal feature
        x, _ = self.encoder(torch.cat([sta_x, tem_x], dim=-1))
        # decoder
        # 5. static enrichment -> enriched temporal feature
        x = self.static_enricher(torch.cat([sta_x, x], dim=-1))
        # 6. multi-head attention decoder ->
        heat_x = self.heat_decoder_seq(x)
        cool_x = self.cool_decoder_seq(x)
        # 7. linear layer -> output
        heat_hat = self.heat_decoder(heat_x)
        cool_hat = self.cool_decoder(cool_x)
        # 8. linear layer -> output
        heat_prob = self.heat_tigger(heat_x)
        cool_prob = self.cool_trigger(cool_x)

        return heat_hat, cool_hat, heat_prob, cool_prob
    
    def criterion(self, heat_hat, cool_hat, heat_prob, cool_prob, y):
        h_mask = (y[:, :, 0] != self.h_zero)
        c_mask = (y[:, :, 1] != self.c_zero)
        heat_prob_loss = nn.functional.binary_cross_entropy_with_logits(
            heat_prob, h_mask.unsqueeze(2).float())
        cool_prob_loss = nn.functional.binary_cross_entropy_with_logits(
            cool_prob, c_mask.unsqueeze(2).float())
        heat_loss = nn.functional.mse_loss(
            heat_hat, y[:, :, 0].unsqueeze(2))
        cool_loss = nn.functional.mse_loss(
            cool_hat, y[:, :, 1].unsqueeze(2))
        return heat_loss, cool_loss, heat_prob_loss, cool_prob_loss
    
    def inverse_transform_load(self, heat, cool):
        return heat * self.scaling.H_STD + self.scaling.H_MEAN, cool * self.scaling.C_STD + self.scaling.C_MEAN

    def generate_load(self, heat_hat, cool_hat, heat_prob, cool_prob):
        heat_hat, cool_hat = self.inverse_transform_load(heat_hat, cool_hat)
        heat_hat[(torch.sigmoid(heat_prob) < self.h_trigger_ratio).squeeze()] = 0
        cool_hat[(torch.sigmoid(cool_prob) < self.c_trigger_ratio).squeeze()] = 0
        return heat_hat, cool_hat
    
    def log_load_difference(self, y, heat_hat, cool_hat, heat_prob, cool_prob, stage):
        heat_true, cool_true = self.inverse_transform_load(
            y[:, :, 0], y[:, :, 1])
        heat_hat, cool_hat = self.generate_load(
            heat_hat, cool_hat, heat_prob, cool_prob)
        heat_diff = nn.functional.l1_loss(heat_true.unsqueeze(2), heat_hat)
        cool_diff = nn.functional.l1_loss(cool_true.unsqueeze(2), cool_hat)
        self.log_dict({f'{stage}/heat_diff': heat_diff, f'{stage}/cool_diff': cool_diff,
                      f'{stage}/total_diff': heat_diff + cool_diff})
        
    def split_reshape(self, batch):
        return (batch['s_cont'], batch['k_cont']), batch['target']
    
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
        self.record_cls_output(y, torch.sigmoid(
            heat_prob), torch.sigmoid(cool_prob))

        self.log_dict({"val/heat_loss": heat_loss, "val/cool_loss": cool_loss,
                       "val/heat_prob_loss": heat_prob_loss, "val/cool_prob_loss": cool_prob_loss,
                       "val/total_loss": loss, "global_step": self.global_step})
        # log actual load diff
        self.log_load_difference(y, heat_hat, cool_hat, heat_prob, cool_prob, stage='val')
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
        self.log_load_difference(y, heat_hat, cool_hat, heat_prob, cool_prob, stage='test')
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
