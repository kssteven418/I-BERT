import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Linear as _linear
from torch.nn import Embedding as _Embedding
from torch.nn import Module, Parameter
from .quant_utils import *

from fairseq.modules import LayerNorm
from fairseq import utils

import logging

logger = logging.getLogger(__name__)

class QuantEmbedding(Module):
    """
    Class to quantize given Embedding layer

    Parameters:
    activation_bit : int
        Bitwidth for quantized weights.
    is_positional : bool, default False
        If the given Embedding layer is positional embedding.
    momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """
    def __init__(self,
                 weight_bit,
                 is_positional=False,
                 momentum=0.95,
                 quant_mode='none'):
        super(QuantEmbedding, self).__init__()

        self.weight_bit = weight_bit
        self.momentum = momentum
        self.quant_mode = quant_mode
        self.per_channel = False
        self.percentile_mode = False
        self.is_positional = is_positional

        if self.quant_mode == "none":
            self.weight_function = None
        elif self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))
                 
    def set_param(self, embedding):
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse
        self.weight = embedding.weight

        if not self.per_channel:
            dim_scaling_factor = 1
        else:
            dim_scaling_factor = self.embedding_dim
        self.register_buffer('weight_scaling_factor', torch.zeros(dim_scaling_factor))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))

        if self.is_positional:
            if self.padding_idx is not None:
                self.max_positions = self.num_embeddings - self.padding_idx - 1
            else:
                self.max_positions = self.num_embeddings


    def forward(self, x, positions=None, incremental_state=None):
        if self.quant_mode == 'none':
            return F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ), None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=0, keepdim=True, out=None)
            w_max, _ = torch.max(w_transform, dim=0, keepdim=True, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        self.weight_scaling_factor = symmetric_linear_quantization_params(
                    self.weight_bit, w_min, w_max, self.per_channel)
        self.weight_integer = self.weight_function(
                    self.weight, self.weight_bit, self.percentile_mode, 
                    self.weight_scaling_factor)

        if self.is_positional:
            assert (positions is None) or (
                self.padding_idx is None
            ), "If positions is pre-computed then padding_idx should not be set."

            if positions is None:
                if incremental_state is not None:
                    # positions is the same for every token when decoding a single step
                    # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                    positions = torch.zeros(
                        (1, 1), device=x.device, dtype=x.dtype
                    ).fill_(int(self.padding_idx + x.size(1)))
                else:
                    positions = utils.make_positions(
                        x, self.padding_idx, onnx_trace=False
                    )
            x = positions

        emb_int = F.embedding(
            x,
            self.weight_integer,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return emb_int * self.weight_scaling_factor, self.weight_scaling_factor


# The input quantization needs to use symmetric quantization!

class QuantAct(Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """
    def __init__(self,
                 activation_bit,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 channel_len=None,
                 quant_mode="none"):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.percentile = False

        if not per_channel:
            self.register_buffer('x_min', torch.zeros(1))
            self.register_buffer('x_max', torch.zeros(1))
            self.register_buffer('act_scaling_factor', torch.zeros(1))
        else:
            assert channel_len is not None
            self.register_buffer('x_min', torch.zeros(channel_len))
            self.register_buffer('x_max', torch.zeros(channel_len))
            self.register_buffer('act_scaling_factor', torch.zeros(channel_len))

        self.quant_mode = quant_mode
        self.per_channel = per_channel

        if self.quant_mode == "none":
            self.act_function = None
        elif self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode: {2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(), self.x_max.item())
    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False
        
    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, x, 
                pre_act_scaling_factor=None, 
                identity=None, 
                identity_scaling_factor=None,
                specified_min=None,
                specified_max=None):
        # collect runnng stats
        x_act = x if identity is None else identity + x
        if self.running_stat:
            if not self.percentile:
                if not self.per_channel:
                    x_min = x_act.data.min()
                    x_max = x_act.data.max()
                else:
                    x_min = x_act.data.min(axis=0).values.min(axis=0).values
                    x_max = x_act.data.max(axis=0).values.max(axis=0).values
            else:
                raise NotImplementedError("percentile mode is not currently supported.")

            # Initialization
            if torch.eq(self.x_min, self.x_max).all():
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum +\
                        x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum +\
                        x_max * (1 - self.act_range_momentum)

        if self.quant_mode == 'none':
            return x_act, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)
        
        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, 
            per_channel=self.per_channel)

        if pre_act_scaling_factor is None:
            # this is for the input quantization 
            quant_act_int = self.act_function(x, self.activation_bit, \
                    self.percentile, self.act_scaling_factor)
        else:
            quant_act_int = fixedpoint_mul.apply(
                    x, pre_act_scaling_factor, 
                    self.activation_bit, self.quant_mode, 
                    self.act_scaling_factor, 
                    identity, identity_scaling_factor)

        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor


class QuantLinear(Module):
    """
    Class to quantize weights of given Linear layer
    
    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 per_channel=False,
                 quant_mode='none'):
        super(QuantLinear, self).__init__()
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.percentile_mode = False

        if self.quant_mode == "none":
            pass
        elif self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if self.quant_mode == 'none':
            return F.linear(x, weight=self.weight, bias=self.bias), None

    	# x / prev_act_scaling_factor = int
        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        # assert that prev_act_scaling_factor is a scalar tensor
        # e.g. all input tensors have the same scalar factor
        assert prev_act_scaling_factor is not None and \
              prev_act_scaling_factor.shape == (1,) 

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, w_min, w_max, self.per_channel)
        self.weight_integer = self.weight_function(
                self.weight, self.weight_bit, self.percentile_mode, 
                self.fc_scaling_factor)

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        self.bias_integer = self.weight_function(self.bias, 
                self.bias_bit, False, bias_scaling_factor)

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
                * bias_scaling_factor, bias_scaling_factor


class IntLayerNorm(Module):
    """
    Class to quantize given LayerNorm layer

    Parameters:
    ----------
    output_bit : int
        Bitwidth for the LayerNorm output.
    overflow_handling : bool, default True
        Whether to do overflow handling if the intermediate values are larger than 32-bit.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize LayerNorm if either 'layernorm' or 'nonlinear' is given.
    """
    def __init__(self,
                 output_bit,
                 overflow_handling=True,
                 quant_mode='none',
                 force_dequant='none'):
        super(IntLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'layernorm']:
            logger.info("Force dequantize layernorm")
            self.quant_mode = 'none'
        self.overflow_handling = overflow_handling
        self.register_buffer('shift', torch.zeros(1))
        self.output_bit = output_bit
        self.dim_sqrt = None

        self.activation = QuantAct(output_bit, quant_mode=self.quant_mode)
        if self.quant_mode == "none":
            pass
        elif quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def fix(self):
        self.overflow_handling = False

    def unfix(self):
        self.overflow_handling = True

    def set_param(self, ln):
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.weight = Parameter(ln.weight.data.clone())
        self.bias = Parameter(ln.bias.data.clone())

    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int ** 2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**32)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logger.info("Dynamic shift adjustment: {} -> {}".format(
                int(shift_old), int(self.shift)))

    def overflow_fallback(self, y_int):
        self.set_shift(y_int)
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift)
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int

    def forward(self, x, scaling_factor=None, exponents=None):
        if self.quant_mode == 'none':
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y ** 2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float) # feature dim(768)
            self.dim_sqrt = torch.sqrt(n).cuda()

        # Normalization: computes mean and variance(std)
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = floor_ste.apply(y_int / 2 ** self.shift) # avoid overflow
        y_sq_int = y_int_shifted ** 2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        
        # overflow handling in training stage
        if self.overflow_handling:
            if var_int.max() >= 2**32:
                var_int = self.overflow_fallback(y_int)
                assert var_int.max() < 2**32
        
        # To be replaced with integer-sqrt kernel that produces the same output
        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2 ** self.shift 
        factor = floor_ste.apply(2**31 / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # scaling and shifting
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        return x, scaling_factor


class IntGELU(Module):
    """
    Class to quantize given GELU layer

    Parameters:
    ----------
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize GELU if either 'gelu' or 'nonlinear' is given.
    """
    def __init__(self,
                 quant_mode='none',
                 force_dequant='none'):
        super(IntGELU, self).__init__()
        self.register_buffer('input_scaling_factor', torch.ones(1))
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'gelu']:
            logger.info("Force dequantize gelu")
            self.quant_mode = 'none'


        if self.quant_mode == 'none':
            self.activation_fn = nn.GELU()
        elif self.quant_mode == 'symmetric':
            pass
        elif quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

        self.k = 1.4142
        self.n = 14 # sufficiently large integer
        self.coeff = [-0.2888, -1.769, 1] # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_erf(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coeff[1] / scaling_factor)
            c_int = torch.floor(self.coeff[2] / scaling_factor ** 2)

        with torch.no_grad():
            sign = torch.sign(x_int)
        abs_int = torch.abs(x_int)
        abs_int = torch.min(abs_int, -b_int)
        y_int = (abs_int + b_int) ** 2 + c_int
        y_int = sign * y_int
        scaling_factor = scaling_factor ** 2 * self.coeff[0]
        y_int = floor_ste.apply(y_int / 2 ** self.n)
        scaling_factor = scaling_factor * 2 ** self.n
        
        return y_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        if self.quant_mode == 'none':
            return self.activation_fn(x), None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)

        shift_int = torch.floor(1. / sigmoid_scaling_factor)

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor


class IntSoftmax(Module):
    """
    Class to quantize given Softmax layer

    Parameters:
    ----------
    output_bit : int
        Bitwidth for the Softmax output.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    force_dequant : str, default 'none'
        Force dequantize Softmax if either 'softmax' or 'nonlinear' is given.
    """
    def __init__(self,
                 output_bit,
                 quant_mode='none',
                 force_dequant='none'):
        super(IntSoftmax, self).__init__()
        self.output_bit = output_bit
        self.quant_mode = quant_mode
        if force_dequant in ['nonlinear', 'softmax']:
            logger.info("Force dequantize softmax")
            self.quant_mode = 'none'


        self.act = QuantAct(16, quant_mode=self.quant_mode)
        self.x0 = -0.6931 # -ln2
        self.n = 30 # sufficiently large integer
        self.coef = [0.35815147, 0.96963238, 1.] # ax**2 + bx + c
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    def fix(self):
        pass

    def unfix(self):
        pass

    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor ** 2)
        z = x_int + b_int
        z = x_int * z
        z = z + c_int
        scaling_factor = self.coef[0] * scaling_factor ** 2
        return z, scaling_factor

    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)
        x_int = torch.max(x_int, self.n * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.n - q)), min=0)
        scaling_factor = exp_scaling_factor / 2 ** self.n
        return exp_int, scaling_factor

    def forward(self, x, scaling_factor):
        if self.quant_mode == 'none':
            return utils.softmax(x, dim=-1, onnx_trace=False), None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        x_int = x / scaling_factor

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max


        exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)
        exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
        exp_int = exp / exp_scaling_factor
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)

        factor = floor_ste.apply(2**32 / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (32 - self.output_bit))
        scaling_factor = 1 / 2 ** self.output_bit
        return exp_int * scaling_factor, scaling_factor
