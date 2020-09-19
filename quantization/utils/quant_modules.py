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
                 signed=True):
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

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('act_scaling_factor', torch.zeros(1))

        self.quant_mode = quant_mode

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

    def forward(self, x, 
                pre_act_scaling_factor=None, 
                identity=None, 
                identity_scaling_factor=None):
        # collect runnng stats
        if self.running_stat:
            if not self.percentile:
                x_min = x.data.min()
                x_max = x.data.max()
            elif self.quant_mode == 'symmetric':
                x_min, x_max = get_percentile_min_max(x.detach().view(-1), 
                                0.1, self.percentage, output_tensor=True)
            elif self.quant_mode == 'asymmetric':
                x_min, x_max = get_percentile_min_max(x.detach().view(-1), 
                                0, self.percentage, output_tensor=True)
            # Initialization
            if self.x_min == self.x_max:
                self.x_min += x_min
                self.x_max += x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = min(self.x_min, x_min)
                self.x_max = max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum +\
                        x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum +\
                        x_max * (1 - self.act_range_momentum)

        if self.quant_mode == 'none':
            return x, None

        # scaling factor and zero point(if necessary) of the activation outputs
        if self.quant_mode == 'symmetric':
            self.act_scaling_factor = symmetric_linear_quantization_params(
                    self.activation_bit, self.x_min, self.x_max, 
                    per_channel=False)
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
            if self.quantize_bias:
                b_min = self.bias.data
                b_max = self.bias.data
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)
            if self.quantize_bias:
                b_min = self.bias.data.min()
                b_max = self.bias.data.max()

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
                 weight_bit,
                 bias_bit,
                 quant_mode='none'):
        super(QuantLayerNorm, self).__init__()
        self.quant_mode = quant_mode
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit

    def set_param(self, ln):
        self.normalized_shape = ln.normalized_shape
        self.eps = ln.eps
        self.weight = Parameter(ln.weight.data.clone())
        self.bias = Parameter(ln.bias.data.clone())
        #self.ln = ln

    def forward(self, x, scaling_factor=None):
        if self.quant_mode == 'none':
            mean = x.mean(axis=2, keepdim=True)
            x = x - mean
            var = torch.mean(x ** 2, axis=2, keepdim=True)
            x = x / torch.sqrt(self.eps + var)
            x = x * self.weight + self.bias
            return x
            #return self.ln(x)
        else:
            return self.ln(x)
            raise NotImplementedError


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
            x, scale_factor = self.prev_act(x)
            x, scale_factor = self.linear(x, scale_factor)
            return x
        else:
            raise NotImplementedError



