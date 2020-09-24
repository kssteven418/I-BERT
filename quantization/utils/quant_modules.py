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

# The input quantization needs to use symmetric quantization!
class QuantAct(Module):
    def __init__(self,
                 activation_bit,
                 act_range_momentum=0.95,
                 running_stat=True,
                 quant_mode="asymmetric",
                 show_flag=False,
                 percentile=False,
                 signed=True,
                 per_channel=False,
                 exponential_quant=False,
                 channel_len=None):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.show_flag = show_flag
        self.percentile = percentile
        self.signed = signed
        self.iter_counter = 0
        self.percentage = 99.9

        if not per_channel:
            self.register_buffer('x_min', torch.zeros(1))
            self.register_buffer('x_max', torch.zeros(1))
            self.register_buffer('act_scaling_factor', torch.zeros(1))
        else:
            assert channel_len is not None
            self.register_buffer('x_min', torch.zeros(channel_len))
            self.register_buffer('x_max', torch.zeros(channel_len))
            self.register_buffer('act_scaling_factor', torch.zeros(channel_len))
            if exponential_quant:
                self.register_buffer('exponents', torch.zeros(channel_len))
                self.register_buffer('global_scaling_factor', torch.zeros(channel_len))


        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.exponential_quant = exponential_quant

        if quant_mode == "none":
            self.act_function = None
        elif quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            # self.act_function = SymmetricQuantFunction.apply
            self.act_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

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
        self.show_flag = True
        
    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True
        self.show_flag = False

    def compute_exponents(self, x_min, x_max):
        if self.running_stat:
            with torch.no_grad():
                n = 2 ** (self.activation_bit - 1) - 1
                self.exponents = torch.zeros_like(self.exponents)

                x_range, _ = torch.max(torch.stack([x_min.abs(), x_max.abs()], dim=1), dim=1)
                x_range = torch.clamp(x_range, min=1e-8)
                x_range_min = x_range.min()
                x_range = x_range / x_range_min

                self.exponents = x_range.log2().ceil()
                self.global_scaling_factor = x_range_min / n

        return self.global_scaling_factor * (2 ** self.exponents)


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
            '''
            elif self.quant_mode == 'symmetric':
                x_min, x_max = get_percentile_min_max(x_act.detach().view(-1), 
                                0.1, self.percentage, output_tensor=True)
            elif self.quant_mode == 'asymmetric':
                x_min, x_max = get_percentile_min_max(x_act.detach().view(-1), 
                                0, self.percentage, output_tensor=True)
            '''
            # Initialization
            #if self.x_min == self.x_max:
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
        
        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max
        # scaling factor and zero point(if necessary) of the activation outputs
        if self.quant_mode == 'symmetric':
            if self.exponential_quant:
                self.act_scaling_factor = self.compute_exponents(x_min, x_max)
            else:
                self.act_scaling_factor = symmetric_linear_quantization_params(
                    self.activation_bit, x_min, x_max, 
                    per_channel=self.per_channel)
        else:
            '''
            self.act_scaling_factor, self.act_zero_point = \
                    asymmetric_linear_quantization_params(self.activation_bit, 
                            self.x_min, self.x_max, 
                            integral_zero_point=True, 
                            signed=self.signed)
            '''
            # TODO Sehoon open up this path once 
            # asymmetric_linear_quantization_params is implemented
            raise NotImplementedError

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

        if self.exponential_quant:
            return quant_act_int * correct_output_scale, self.global_scaling_factor, self.exponents

        return quant_act_int * correct_output_scale, self.act_scaling_factor

#class QuantSoftmax(Module):
#    def __init__

class QuantLinear(Module):
    """
    Class to quantize given linear layer weights
    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 quant_mode='none',
                 per_channel=False,
                 show_flag=False,
                 weight_percentile=False,
                 save_path=None,
                 threshold=None):
        """
        weight: bit-setting for weight
        running_stat: determines whether the activation range is updated or froze
        """
        super(QuantLinear, self).__init__()
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.show_flag = show_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.save_path = save_path
        self.counter = 0
        self.checkpoint_iter_threshold = threshold

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply

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
        self.register_buffer('fc_zero_point', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('bias_integer', torch.zeros_like(self.weight))
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None

    def fix(self):
        self.show_flag = True

    def unfix(self):
        self.show_flag = False

    # prev_act_scaling_factor: used to scaling the bias term
    # also, x / prev_act_scaling_factor = int
    def forward(self, x, prev_act_scaling_factor=None, prev_act_zero_point=None):
        """
        using quantized weights to forward activation x
        """
        if self.quant_mode == 'none':
            return F.linear(x, weight=self.weight, bias=self.bias), None

        # assert that prev_act_scaling_factor is a scalar tensor
        # e.g. all input tensors have the same scalar factor
        assert prev_act_scaling_factor is not None and \
              prev_act_scaling_factor.shape == (1,) 

        #print('x shape @ QuantLinear', x.shape)

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # we need to add asymmetric here later, for now just ignore it
        if self.quant_mode == 'symmetric':
            # TODO: for now, we alway enable fraction number as well as make denom=10250, we can make it more auto later.
            self.fc_scaling_factor = symmetric_linear_quantization_params(
                    self.weight_bit, w_min, w_max, self.per_channel)
            self.weight_integer = self.weight_function(
                    self.weight, self.weight_bit, self.weight_percentile, 
                    self.fc_scaling_factor)

            # fc_scaling_factor is per_channel  2 x n 
            # prev_act_scaling_factor: 2
            #print('fc shape @ QuantLinear', self.fc_scaling_factor.shape)
            #print('prev act shape @ QuantLinear', prev_act_scaling_factor.shape)
            bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

            self.bias_integer = self.weight_function(self.bias, 
                    self.bias_bit, False, bias_scaling_factor)
            #print('bias integer', self.bias_integer.shape)
            #print('bias scaling factor', bias_scaling_factor.shape)
        else:
            raise Exception('For weight, we only support symmetric quantization.')

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        #print('x_int shape @ QuantLinear', x_int.shape)
        #print('weight shape @ QuantLinear', self.weight_integer.shape)
        #print('bias scaling factor shape @ QuantLinear', bias_scaling_factor[0].shape)
        #print('bias scaling factor shape 2 @ QuantLinear', bias_scaling_factor.shape)

        self.counter += 1

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
                * bias_scaling_factor, bias_scaling_factor

class QuantLayerNorm(Module):
    def __init__(self,
                 #weight_bit,
                 #bias_bit,
                 output_bit,
                 quant_mode='none'):
        super(QuantLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        #self.weight_bit = weight_bit
        #self.bias_bit = bias_bit
        self.output_bit = output_bit

        self.activation = QuantAct(output_bit, quant_mode=self.quant_mode)
        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply

    def set_param(self, ln):
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.weight = Parameter(ln.weight.data.clone())
        self.bias = Parameter(ln.bias.data.clone())
        self.shift = 5

    def forward(self, x, scaling_factor=None, exponents=None):
        if exponents is None:
        #if self.quant_mode == 'none':
            
            mean = x.mean(axis=2, keepdim=True)
            y = x - mean
            var = torch.mean(y ** 2, axis=2, keepdim=True)
            x = y / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x, None

        elif self.quant_mode == 'symmetric':
            n = torch.tensor(x.shape[2], dtype=torch.float) # 768, feature dim
            x_int = x / scaling_factor
            mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
            y_int = x_int - mean_int
            y_sq_int = y_int ** 2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            if y_sq_int.max() > 2**32:
                print(y_sq_int.max())
            scaling_factor = 1 / torch.sqrt(var_int) # cast to float
            scaling_factor = scaling_factor * torch.sqrt(n)
            x = y_int * scaling_factor

            if self.quant_mode == 'symmetric':
                bias = self.bias.data.detach() / self.weight.data.detach()
                bias_scaling_factor = bias.clone() # bias_int == torch.ones_like(bias)
            else:
                raise Exception('For LN, we only support symmetric quantization.')

            # requantize here?
            x, scaling_factor = self.activation(x, scaling_factor,
                               identity=bias,
                               identity_scaling_factor=bias_scaling_factor)

            x = x * self.weight

            return x, scaling_factor * self.weight


class QuantLinearWrapper(Module):
    """
    Wrapper class for QuantLinear module.
    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 quant_mode='none',
                 per_channel=False,
                 show_flag=False,
                 weight_percentile=False,
                 save_path=None,
                 threshold=None):
        super(QuantLinearWrapper, self).__init__()
        self.quant_mode = quant_mode
        self.prev_act = QuantAct(weight_bit, quant_mode=self.quant_mode)
        self.linear = QuantLinear(weight_bit, bias_bit,
                quant_mode, per_channel, show_flag, weight_percentile,
                save_path, threshold)

    def set_param(self, linear):
        self.linear.set_param(linear)

    def forward(self, x):
        if self.quant_mode == 'symmetric':
            x, scaling_factor = self.prev_act(x)
            x, scaling_factor = self.linear(x, scaling_factor)
            return x
        else:
            raise NotImplementedError



