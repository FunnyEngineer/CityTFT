# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from typing import Dict, Tuple, Optional, List

MAKE_CONVERT_COMPATIBLE = os.environ.get("TFT_SCRIPTING", None) is not None


class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden_size, eps=eps)

    def forward(self, x):
        return self.ln(x)


class GLU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class GRN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 context_hidden_size=None,
                 dropout=0.0,):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        else:
            self.lin_c = nn.Identity()
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if self.out_proj is None else self.out_proj(a)
        x = x + y
        return self.layer_norm(x)


# @torch.jit.script #Currently broken with autocast
def fused_pointwise_linear_v1(x, a, b):
    out = torch.mul(x.unsqueeze(-1), a)
    out = out + b
    return out


@torch.jit.script
def fused_pointwise_linear_v2(x, a, b):
    out = x.unsqueeze(3) * a
    out = out + b
    return out


class TFTEmbedding(nn.Module):
    def __init__(self, config, initialize_cont_params=True):
        # initialize_cont_params=False prevents form initializing parameters inside this class
        # so they can be lazily initialized in LazyEmbedding module
        super().__init__()
        self.s_cat_inp_lens = config.static_categorical_inp_lens
        self.t_cat_k_inp_lens = config.temporal_known_categorical_inp_lens
        self.t_cat_o_inp_lens = config.temporal_observed_categorical_inp_lens
        self.s_cont_inp_size = config.static_continuous_inp_size
        self.t_cont_k_inp_size = config.temporal_known_continuous_inp_size
        self.t_cont_o_inp_size = config.temporal_observed_continuous_inp_size
        self.t_tgt_size = config.temporal_target_size

        self.hidden_size = config.hidden_size

        # There are 7 types of input:
        # 1. Static categorical
        # 2. Static continuous
        # 3. Temporal known a priori categorical
        # 4. Temporal known a priori continuous
        # 5. Temporal observed categorical
        # 6. Temporal observed continuous
        # 7. Temporal observed targets (time series obseved so far)

        self.s_cat_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.s_cat_inp_lens]) if self.s_cat_inp_lens else None
        self.t_cat_k_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_k_inp_lens]) if self.t_cat_k_inp_lens else None
        self.t_cat_o_embed = nn.ModuleList([
            nn.Embedding(n, self.hidden_size) for n in self.t_cat_o_inp_lens]) if self.t_cat_o_inp_lens else None

        if initialize_cont_params:
            self.s_cont_embedding_vectors = nn.Parameter(torch.Tensor(
                self.s_cont_inp_size, self.hidden_size)) if self.s_cont_inp_size else None
            self.t_cont_k_embedding_vectors = nn.Parameter(torch.Tensor(
                self.t_cont_k_inp_size, self.hidden_size)) if self.t_cont_k_inp_size else None
            self.t_cont_o_embedding_vectors = nn.Parameter(torch.Tensor(
                self.t_cont_o_inp_size, self.hidden_size)) if self.t_cont_o_inp_size else None
            self.t_tgt_embedding_vectors = nn.Parameter(
                torch.Tensor(self.t_tgt_size, self.hidden_size))

            self.s_cont_embedding_bias = nn.Parameter(torch.zeros(
                self.s_cont_inp_size, self.hidden_size)) if self.s_cont_inp_size else None
            self.t_cont_k_embedding_bias = nn.Parameter(torch.zeros(
                self.t_cont_k_inp_size, self.hidden_size)) if self.t_cont_k_inp_size else None
            self.t_cont_o_embedding_bias = nn.Parameter(torch.zeros(
                self.t_cont_o_inp_size, self.hidden_size)) if self.t_cont_o_inp_size else None
            self.t_tgt_embedding_bias = nn.Parameter(torch.zeros(self.t_tgt_size, self.hidden_size))

            self.reset_parameters()

    def reset_parameters(self):
        if self.s_cont_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.s_cont_embedding_vectors)
            torch.nn.init.zeros_(self.s_cont_embedding_bias)
        if self.t_cont_k_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_cont_k_embedding_vectors)
            torch.nn.init.zeros_(self.t_cont_k_embedding_bias)
        if self.t_cont_o_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_cont_o_embedding_vectors)
            torch.nn.init.zeros_(self.t_cont_o_embedding_bias)
        if self.t_tgt_embedding_vectors is not None:
            torch.nn.init.xavier_normal_(self.t_tgt_embedding_vectors)
            torch.nn.init.zeros_(self.t_tgt_embedding_bias)
        if self.s_cat_embed is not None:
            for module in self.s_cat_embed:
                module.reset_parameters()
        if self.t_cat_k_embed is not None:
            for module in self.t_cat_k_embed:
                module.reset_parameters()
        if self.t_cat_o_embed is not None:
            for module in self.t_cat_o_embed:
                module.reset_parameters()

    def _apply_embedding(self,
                         cat: Optional[Tensor],
                         cont: Optional[Tensor],
                         cat_emb: Optional[nn.ModuleList],
                         cont_emb: Tensor,
                         cont_bias: Tensor,
                         ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        e_cat = torch.stack([embed(cat[..., i])
                            for i, embed in enumerate(cat_emb)], dim=-2) if cat is not None else None
        if cont is not None:
            # the line below is equivalent to following einsums
            # e_cont = torch.einsum('btf,fh->bthf', cont, cont_emb)
            # e_cont = torch.einsum('bf,fh->bhf', cont, cont_emb)
            if MAKE_CONVERT_COMPATIBLE:
                e_cont = torch.mul(cont.unsqueeze(-1), cont_emb)
                e_cont = e_cont + cont_bias
            else:
                e_cont = fused_pointwise_linear_v1(cont, cont_emb, cont_bias)
        else:
            e_cont = None

        if e_cat is not None and e_cont is not None:
            return torch.cat([e_cat, e_cont], dim=-2)
        elif e_cat is not None:
            return e_cat
        elif e_cont is not None:
            return e_cont
        else:
            return None

    def forward(self, x: Dict[str, Tensor]):
        # temporal/static categorical/continuous known/observed input
        s_cat_inp = x.get('s_cat', None)
        s_cont_inp = x.get('s_cont', None)
        t_cat_k_inp = x.get('k_cat', None)
        t_cont_k_inp = x.get('k_cont', None)
        t_cat_o_inp = x.get('o_cat', None)
        t_cont_o_inp = x.get('o_cont', None)
        t_tgt_obs = x['target']  # Has to be present

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        s_cat_inp = s_cat_inp[:, 0, :] if s_cat_inp.shape[-1] else None
        s_cont_inp = s_cont_inp[:, 0, :] if s_cont_inp.shape[-1] else None

        s_inp = self._apply_embedding(s_cat_inp,
                                      s_cont_inp,
                                      self.s_cat_embed,
                                      self.s_cont_embedding_vectors,
                                      self.s_cont_embedding_bias)

        # TODO: preprocessing t_cat_k_inp, t_cont_k_inp
        t_cat_k_inp = t_cat_k_inp if t_cat_k_inp.nelement() else None
        t_cont_k_inp = t_cont_k_inp if t_cont_k_inp.nelement() else None

        t_known_inp = self._apply_embedding(t_cat_k_inp,
                                            t_cont_k_inp,
                                            self.t_cat_k_embed,
                                            self.t_cont_k_embedding_vectors,
                                            self.t_cont_k_embedding_bias)
        t_cat_o_inp = t_cat_o_inp if t_cat_o_inp.nelement() else None
        t_cont_o_inp = t_cont_o_inp if t_cont_o_inp.nelement() else None

        t_observed_inp = self._apply_embedding(t_cat_o_inp,
                                               t_cont_o_inp,
                                               self.t_cat_o_embed,
                                               self.t_cont_o_embedding_vectors,
                                               self.t_cont_o_embedding_bias)

        # Temporal observed targets
        # t_observed_tgt = torch.einsum('btf,fh->btfh', t_tgt_obs, self.t_tgt_embedding_vectors)
        # TODO: turn v2 into v1
        if MAKE_CONVERT_COMPATIBLE:
            t_observed_tgt = torch.matmul(t_tgt_obs.unsqueeze(3).unsqueeze(
                4), self.t_tgt_embedding_vectors.unsqueeze(1)).squeeze(3)
            t_observed_tgt = t_observed_tgt + self.t_tgt_embedding_bias
        else:
            t_observed_tgt = fused_pointwise_linear_v2(
                t_tgt_obs, self.t_tgt_embedding_vectors, self.t_tgt_embedding_bias)

        return s_inp, t_known_inp, t_observed_inp, t_observed_tgt


class LazyEmbedding(nn.modules.lazy.LazyModuleMixin, TFTEmbedding):
    cls_to_become = TFTEmbedding

    def __init__(self, config):
        super().__init__(config, initialize_cont_params=False)

        if config.static_continuous_inp_size:
            self.s_cont_embedding_vectors = UninitializedParameter()
            self.s_cont_embedding_bias = UninitializedParameter()
        else:
            self.s_cont_embedding_vectors = None
            self.s_cont_embedding_bias = None

        if config.temporal_known_continuous_inp_size:
            self.t_cont_k_embedding_vectors = UninitializedParameter()
            self.t_cont_k_embedding_bias = UninitializedParameter()
        else:
            self.t_cont_k_embedding_vectors = None
            self.t_cont_k_embedding_bias = None

        if config.temporal_observed_continuous_inp_size:
            self.t_cont_o_embedding_vectors = UninitializedParameter()
            self.t_cont_o_embedding_bias = UninitializedParameter()
        else:
            self.t_cont_o_embedding_vectors = None
            self.t_cont_o_embedding_bias = None
        self.t_tgt_embedding_vectors = UninitializedParameter()
        self.t_tgt_embedding_bias = UninitializedParameter()

    def initialize_parameters(self, x):
        if self.has_uninitialized_params():
            s_cont_inp = x.get('s_cont', None)
            t_cont_k_inp = x.get('k_cont', None)
            t_cont_o_inp = x.get('o_cont', None)
            t_tgt_obs = x['target']  # Has to be present

            if s_cont_inp.shape[-1]:
                self.s_cont_embedding_vectors.materialize((s_cont_inp.shape[-1], self.hidden_size))
                self.s_cont_embedding_bias.materialize((s_cont_inp.shape[-1], self.hidden_size))

            if t_cont_k_inp.shape[-1]:
                self.t_cont_k_embedding_vectors.materialize(
                    (t_cont_k_inp.shape[-1], self.hidden_size))
                self.t_cont_k_embedding_bias.materialize((t_cont_k_inp.shape[-1], self.hidden_size))

            if t_cont_o_inp.shape[-1]:
                self.t_cont_o_embedding_vectors.materialize(
                    (t_cont_o_inp.shape[-1], self.hidden_size))
                self.t_cont_o_embedding_bias.materialize((t_cont_o_inp.shape[-1], self.hidden_size))

            self.t_tgt_embedding_vectors.materialize((t_tgt_obs.shape[-1], self.hidden_size))
            self.t_tgt_embedding_bias.materialize((t_tgt_obs.shape[-1], self.hidden_size))

            self.reset_parameters()


class VariableSelectionNetwork(nn.Module):
    def __init__(self, config, num_inputs):
        super().__init__()
        self.joint_grn = GRN(config.hidden_size*num_inputs, config.hidden_size,
                             output_size=num_inputs, context_hidden_size=config.hidden_size)
        self.var_grns = nn.ModuleList(
            [GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(num_inputs)])

    def forward(self, x: Tensor, context: Optional[Tensor] = None):
        Xi = torch.flatten(x, start_dim=-2)
        grn_outputs = self.joint_grn(Xi, c=context)
        sparse_weights = F.softmax(grn_outputs, dim=-1)
        transformed_embed_list = [m(x[..., i, :]) for i, m in enumerate(self.var_grns)]
        transformed_embed = torch.stack(transformed_embed_list, dim=-1)
        # the line below performs batched matrix vector multiplication
        # for temporal features it's bthf,btf->bth
        # for static features it's bhf,bf->bh
        variable_ctx = torch.matmul(transformed_embed, sparse_weights.unsqueeze(-1)).squeeze(-1)

        return variable_ctx, sparse_weights


class StaticCovariateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vsn = VariableSelectionNetwork(config, config.num_static_vars)
        self.context_grns = nn.ModuleList(
            [GRN(config.hidden_size, config.hidden_size, dropout=config.dropout) for _ in range(4)])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        variable_ctx, sparse_weights = self.vsn(x)

        # Context vectors:
        # variable selection context
        # enrichment context
        # state_c context
        # state_h context
        cs, ce, ch, cc = [m(variable_ctx) for m in self.context_grns]

        return cs, ce, ch, cc


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        assert config.hidden_size % config.n_head == 0
        self.d_head = config.hidden_size // config.n_head
        self.qkv_linears = nn.Linear(
            config.hidden_size, (2 * self.n_head + 1) * self.d_head, bias=False)
        self.out_proj = nn.Linear(self.d_head, config.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.out_dropout = nn.Dropout(config.dropout)
        self.scale = self.d_head**-0.5
        self.register_buffer("_mask", torch.triu(torch.full(
            (config.example_length, config.example_length), float('-inf')), 1).unsqueeze(0))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bs, t, h_size = x.shape
        qkv = self.qkv_linears(x)
        q, k, v = qkv.split((self.n_head * self.d_head, self.n_head *
                            self.d_head, self.d_head), dim=-1)
        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)

        # attn_score = torch.einsum('bind,bjnd->bnij', q, k)
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)

        attn_score = attn_score + self._mask

        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)

        # attn_vec = torch.einsum('bnij,bjd->bnid', attn_prob, v)
        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)

        return out, attn_prob


class TFTBack(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder_length = config.encoder_length
        self.history_vsn = VariableSelectionNetwork(config, config.num_historic_vars)
        self.history_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.future_vsn = VariableSelectionNetwork(config, config.num_future_vars)
        self.future_encoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)

        self.input_gate = GLU(config.hidden_size, config.hidden_size)
        self.input_gate_ln = LayerNorm(config.hidden_size, eps=1e-3)

        self.enrichment_grn = GRN(config.hidden_size,
                                  config.hidden_size,
                                  context_hidden_size=config.hidden_size,
                                  dropout=config.dropout)
        self.attention = InterpretableMultiHeadAttention(config)
        self.attention_gate = GLU(config.hidden_size, config.hidden_size)
        self.attention_ln = LayerNorm(config.hidden_size, eps=1e-3)

        self.positionwise_grn = GRN(config.hidden_size,
                                    config.hidden_size,
                                    dropout=config.dropout)

        self.decoder_gate = GLU(config.hidden_size, config.hidden_size)
        self.decoder_ln = LayerNorm(config.hidden_size, eps=1e-3)

        self.quantile_proj = nn.Linear(
            config.hidden_size, config.temporal_target_size * len(config.quantiles))
        self.prob_proj = nn.Linear(config.hidden_size, config.temporal_target_size)

    def forward(self, historical_inputs, cs, ch, cc, ce, future_inputs):
        historical_features, _ = self.history_vsn(historical_inputs, cs)
        history, state = self.history_encoder(historical_features, (ch, cc))
        future_features, _ = self.future_vsn(future_inputs, cs)
        future, _ = self.future_encoder(future_features, state)
        torch.cuda.synchronize()

        # skip connection
        input_embedding = torch.cat([historical_features, future_features], dim=1)
        temporal_features = torch.cat([history, future], dim=1)
        temporal_features = self.input_gate(temporal_features)
        temporal_features = temporal_features + input_embedding
        temporal_features = self.input_gate_ln(temporal_features)

        # Static enrichment
        enriched = self.enrichment_grn(temporal_features, c=ce)

        # Temporal self attention
        x, _ = self.attention(enriched)

        # Don't compute hictorical quantiles
        x = x[:, self.encoder_length:, :]
        temporal_features = temporal_features[:, self.encoder_length:, :]
        enriched = enriched[:, self.encoder_length:, :]

        x = self.attention_gate(x)
        x = x + enriched
        x = self.attention_ln(x)

        # Position-wise feed-forward
        x = self.positionwise_grn(x)

        # Final skip connection
        x = self.decoder_gate(x)
        x = x + temporal_features
        x = self.decoder_ln(x)

        out = self.quantile_proj(x)
        prob_out = F.sigmoid(self.prob_proj(x))

        return out, prob_out


class TemporalFusionTransformer(L.LightningModule):
    """ 
    Implementation of https://arxiv.org/abs/1912.09363 
    """

    def __init__(self, config, scaling):
        super().__init__()

        if hasattr(config, 'model'):
            config = config.model

        # this determines from how distant past we want to use data from
        self.encoder_length = config.encoder_length

        self.embedding = LazyEmbedding(config)
        self.static_encoder = StaticCovariateEncoder(config)

        if MAKE_CONVERT_COMPATIBLE:
            self.TFTpart2 = TFTBack(config)
        else:
            self.TFTpart2 = torch.jit.script(TFTBack(config))
            # self.TFTpart2 = TFTBack(config)

        # for loss function
        self.quantile = config.quantiles
        self.quantile_loss = QuantileLoss(self.quantile)

        self.scaling = scaling
        self.h_zero = -scaling.H_MEAN / scaling.H_STD
        self.c_zero = -scaling.C_MEAN / scaling.C_STD
        self.h_trigger_ratio = scaling.H_TRIGGER_RATIO
        self.c_trigger_ratio = scaling.C_TRIGGER_RATIO

        self.validation_confusion_matrix = {'heat': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                                            'cool': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}
        
        self.save_hyperparameters()
    

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        s_inp, t_known_inp, t_observed_inp, t_observed_tgt = self.embedding(x)

        # Static context
        cs, ce, ch, cc = self.static_encoder(s_inp)
        ch, cc = ch.unsqueeze(0), cc.unsqueeze(0)  # lstm initial states

        # Temporal input
        _historical_inputs = [t_known_inp[:, :self.encoder_length, :]]
        
        if t_observed_inp is not None:
            _historical_inputs.insert(0, t_observed_inp[:, :self.encoder_length, :])

        historical_inputs = torch.cat(_historical_inputs, dim=-2)
        future_inputs = t_known_inp[:, :self.encoder_length]
        return self.TFTpart2(historical_inputs, cs, ch, cc, ce, future_inputs)

    def criterion(self, out, prob_out, targets):
        h_mask = targets[:, :, 0].ne(self.h_zero)
        c_mask = targets[:, :, 1].ne(self.c_zero)
        h_prob_loss = F.binary_cross_entropy(prob_out[:, :, 0].unsqueeze(2), h_mask.unsqueeze(2).float())
        c_prob_loss = F.binary_cross_entropy(prob_out[:, :, 1].unsqueeze(2), c_mask.unsqueeze(2).float())
        heat_loss = self.quantile_loss(out[:, :, :3][h_mask], targets[:, :, 0].unsqueeze(2)[h_mask]).sum()
        cool_loss = self.quantile_loss(out[:, :, 3:][c_mask], targets[:, :, 1].unsqueeze(2)[c_mask]).sum()
        return heat_loss, cool_loss, h_prob_loss, c_prob_loss

    def training_step(self, batch, batch_idx):
        out, prob_out = self(batch)

        targets = batch['target']
        heat_loss, cool_loss, h_prob_loss, c_prob_loss = self.criterion(out, prob_out, targets)
        loss = heat_loss + cool_loss + h_prob_loss + c_prob_loss
        self.log_cls_result(targets, prob_out)

        self.log_dict({"train/heat_loss": heat_loss, "train/cool_loss": cool_loss,
                       "train/heat_prob_loss": h_prob_loss, "train/cool_prob_loss": c_prob_loss,
                       "train/total_loss": loss, "global_step": self.global_step})
        return loss

    def validation_step(self, batch, batch_idx):
        out, prob_out = self(batch)

        targets = batch['target']

        heat_loss, cool_loss, h_prob_loss, c_prob_loss = self.criterion(out, prob_out, targets)
        
        heat_loss = 0 if torch.isnan(heat_loss) else heat_loss
        cool_loss = 0 if torch.isnan(cool_loss) else cool_loss
        loss = heat_loss + cool_loss + h_prob_loss + c_prob_loss
        self.record_cls_output(targets, prob_out)

        self.log_dict({"val/heat_loss": heat_loss, "val/cool_loss": cool_loss,
                       "val/heat_prob_loss": h_prob_loss, "val/cool_prob_loss": c_prob_loss,
                       "val/total_loss": loss, "global_step": self.global_step})
        return loss

    def on_validation_epoch_end(self):
        for h_or_c in ('heat', 'cool'):
            # calculate accuracy, precision, recall, and f1 score
            sin_confu_mat = self.validation_confusion_matrix[h_or_c]
            accuracy = (sin_confu_mat['tp'] + sin_confu_mat['tn']) / (
                sin_confu_mat['tp'] + sin_confu_mat['tn'] + sin_confu_mat['fp'] + sin_confu_mat['fn'])
            precision = sin_confu_mat['tp'] / (sin_confu_mat['tp'] + sin_confu_mat['fp'])
            recall = sin_confu_mat['tp'] / (sin_confu_mat['tp'] + sin_confu_mat['fn'])
            f1 = 2 * precision * recall / (precision + recall)
            # log the results
            self.log_dict({f"val_prob/{h_or_c}/accuracy": accuracy, f"val_prob/{h_or_c}/precision": precision,
                           f"val_prob/{h_or_c}/recall": recall, f"val_prob/{h_or_c}/f1": f1})
        self.validation_confusion_matrix = {'heat': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                                            'cool': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}
    
    def record_cls_output(self, y, prob_out):
        heat_prob = prob_out[..., 0].unsqueeze(2)
        cool_prob = prob_out[..., 1].unsqueeze(2)
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
    
    def test_step(self, batch, batch_idx):
        out, prob_out = self(batch)

        targets = batch['target']

        heat_loss, cool_loss, h_prob_loss, c_prob_loss = self.criterion(out, prob_out, targets)
        
        heat_loss = 0 if torch.isnan(heat_loss) else heat_loss
        cool_loss = 0 if torch.isnan(cool_loss) else cool_loss
        loss = heat_loss + cool_loss + h_prob_loss + c_prob_loss
        self.log_dict({"test/heat_loss": heat_loss, "test/cool_loss": cool_loss,
                       "test/heat_prob_loss": h_prob_loss, "test/cool_prob_loss": c_prob_loss,
                       "test/total_loss": loss, "global_step": self.global_step})
        return loss

    def predict_step(self, batch, batch_idx):
        out, prob_out = self(batch)

        targets = batch['target']

        targets[:, :, 0] = targets[:, :, 0] * self.scaling.H_STD + self.scaling.H_MEAN
        targets[:, :, 1] = targets[:, :, 1] * self.scaling.C_STD + self.scaling.C_MEAN
        return  self.predict(out, prob_out), targets

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def predict(self, out, prob_out):
        heat_prob = prob_out[..., 0].unsqueeze(2)
        cool_prob = prob_out[..., 1].unsqueeze(2)

        out[:, :, :3] = out[:, :, :3] * self.scaling.H_STD + self.scaling.H_MEAN
        out[:, :, 3:] = out[:, :, 3:] * self.scaling.C_STD + self.scaling.C_MEAN
        idx = self.quantile.index(0.5)
        heat_hat = out[:, :, idx]
        cool_hat = out[:, :, idx + 3]
        
        heat_hat[(heat_prob < self.h_trigger_ratio).squeeze()] = 0
        cool_hat[(cool_prob < self.c_trigger_ratio).squeeze()] = 0

        return heat_hat, cool_hat

    def log_cls_result(self, y, prob_out, stage='train'):
        heat_prob = prob_out[..., 0].unsqueeze(2)
        cool_prob = prob_out[..., 1].unsqueeze(2)
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
   