# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout

from quantization.utils.quant_modules import *

class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        quant_mode: str = 'none',
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        self.quant_mode = quant_mode

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)

        self.input_act = QuantAct(8, quant_mode=self.quant_mode)

        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            quant_mode=quant_mode,
        )

        # TODO(Sehoon): proper output bit? 32 or 8
        self.pre_self_attn_layer_norn_act = QuantAct(16, quant_mode=self.quant_mode)

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        #self.self_attn_layer_norm = QuantLayerNorm(8, 32, quant_mode=self.quant_mode)
        #self.self_attn_layer_norm.set_param(self_attn_layer_norm)

        self.fc1_act = QuantAct(8, quant_mode=self.quant_mode)
        self.fc2_act = QuantAct(8, quant_mode=self.quant_mode)
        self.output_act = QuantAct(8, quant_mode=self.quant_mode)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        self.pre_final_layer_norn_act = QuantAct(16, quant_mode=self.quant_mode)
        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        #self.final_layer_norm = QuantLayerNorm(8, 32, quant_mode=self.quant_mode)
        #self.final_layer_norm.set_param(final_layer_norm)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        linear = QuantLinear(8, bias_bit=32, quant_mode=self.quant_mode, per_channel=True)
        linear.set_param(nn.Linear(input_dim, output_dim))
        return quant_noise(linear, q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        linear = QuantLinear(8, bias_bit=32, quant_mode=self.quant_mode, per_channel=True)
        linear.set_param(nn.Linear(input_dim, output_dim))
        return quant_noise(linear, q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
        quant_mode,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            quant_mode=quant_mode,
            return_output_scale=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        x, x_scale_factor = self.input_act(x)
        residual, residual_scale_factor = x, x_scale_factor
        x, x_scale_factor, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            query_scale=x_scale_factor,
            key_scale=x_scale_factor,
            value_scale=x_scale_factor,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)

        # 1st LN
        x, x_scaler_factor = self.pre_self_attn_layer_norn_act(
                x, x_scale_factor,
                identity=residual,
                identity_scaling_factor=residual_scale_factor)

        '''
        # Check LN
        bias = self.self_attn_layer_norm.bias
        weight = self.self_attn_layer_norm.weight
        mean = x.mean(axis=2, keepdim=True)
        var = torch.mean((x-mean)**2, axis=2, keepdim=True)
        y = (x-mean) / torch.sqrt(var + 1e-5)
        y = y * weight + bias
        #print(y[0])
        '''

        x = self.self_attn_layer_norm(x)
        #print(x[0])
        #print(y[0])


        x, scale_factor = self.fc1_act(x)
        residual, residual_scale_factor = x, scale_factor

        x, _ = self.fc1(x, scale_factor)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        x, scale_factor = self.fc2_act(x) #TODO
        x, x_scale_factor = self.fc2(x, scale_factor)
        x = self.dropout_module(x)
        x, x_scaler_factor = self.pre_final_layer_norn_act(
                x, x_scale_factor,
                identity=residual,
                identity_scaling_factor=residual_scale_factor)
        x = self.final_layer_norm(x)
        return x, attn
