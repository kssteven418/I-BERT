import math
import numpy as np
from torch.autograd import Function, Variable
import torch
import bisect
from fractions import Fraction
import decimal
from decimal import Decimal
import time

def bmm_wrapper(input1, scaling_factor1, input2, scaling_factor2):
    input_int1 = input1 / scaling_factor1
    input_int2 = input2 / scaling_factor2
    output_int =torch.bmm(input_int1, input_int2)
    output_scaling_factor = scaling_factor1 * scaling_factor2
    return output_int * output_scaling_factor, output_scaling_factor


def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    input_length = input.shape[0]

    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)

    upper_bound = torch.kthvalue(input, k=upper_index).values

    if lower_percentile == 0:
        lower_bound = upper_bound * 0
        # lower_index += 1
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.
    input: single-precision input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """

    # reshape scale and zeropoint for convolutional weights and activation
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    # quantized = float / scale + zero_point
    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input
    return torch.round(1. / scale * input + zero_point)


def asymmetric_linear_quantization_params(num_bits,
                                          saturation_min,
                                          saturation_max,
                                          integral_zero_point=True,
                                          signed=False):
    """
    Compute the scaling factor and zeropoint with the given quantization range.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """

    # these computation, do not involve any gradient, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2**num_bits - 1
        scale = torch.clamp((saturation_max - saturation_min), min=1e-8) / float(n)

        scale = scale 

        # For this projection, saturation_min = 0 (we only do asymmetric for activatoin.)
        zero_point = -saturation_min / scale

        if integral_zero_point:
            if isinstance(zero_point, torch.Tensor):
                zero_point = zero_point.round()
            else:
                zero_point = float(round(zero_point))
        if signed:
            zero_point += 2**(num_bits - 1)

        return scale, zero_point


class AsymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values with given range and bit-setting.
    Currently only support inference, but have not supported back-propagation.
    """
    @staticmethod
    def forward(ctx, x, k, percentile_mode=False, specified_scale=None, specified_zero_point=None):
        """
        x: single-precision value to be quantized
        k: bit-setting for x
        x_min: lower bound for quantization range
        x_max=None
        """
        if specified_scale is not None:
            scale = specified_scale
        if specified_zero_point is not None:
            zero_point = specified_zero_point
        else:
            zero_point = torch.tensor(0).cuda()

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        n = 2**k - 1

        new_quant_x = torch.clamp(new_quant_x, 0, n)

        ctx.scale = scale

        return new_quant_x

def symmetric_linear_quantization_params(num_bits,
                                        saturation_min,
                                        saturation_max,
                                        per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    """
    # in this part, we do not need any gradient computation,
    # in order to enfore this, we put torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1

        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n 

        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n 

    return scale 

class SymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, percentile_mode=False, specified_scale=None):
        
        if specified_scale is not None:
            scale = specified_scale

        zero_point = torch.tensor(0.).cuda()

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n, n-1)

        ctx.scale = scale 
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None, None

class round_ste(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

# z = w * x
# inputs alpha_x * alpha_w / alpha_z
def batch_frexp(inputs, max_bit=31):
    #
    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)
    
    output_m, output_e = np.frexp(inputs.cpu().numpy())
    # print(output_m * (2 ** 31))
    # output_m = np.round( output_m * (2 ** 31) )
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(Decimal(m * (2**max_bit)).quantize(Decimal('1'), 
                                                               rounding=decimal.ROUND_HALF_UP))
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)
    # output_m = np.array(np.round( np.float64(output_m) * (2 ** 31)))

    # inputs = [output_m*2^32] * 2^output_e / 2^32
    # output_e = np.round(output_e - 31)
    output_e = float(max_bit) - output_e

    return torch.from_numpy( output_m ).cuda().view(shape_of_input), \
           torch.from_numpy( output_e ).cuda().view(shape_of_input)

class fixedpoint_mul(Function):
    @ staticmethod
    def forward (ctx, pre_act, pre_act_scaling_factor, 
                 bit_num, quant_mode, z_scaling_factor, 
                 identity=None, identity_scaling_factor=None):

        #TODO(Sehoon): May require other type of reshape
        if len(pre_act_scaling_factor.shape) == 3:
            reshape = lambda x : x
        else:
            reshape = lambda x : x.view(1, 1, -1)
        ctx.identity = identity

        if quant_mode == 'symmetric':
            n = 2 ** (bit_num - 1) - 1
        else:
            n = 2 ** bit_num - 1

        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor
            
            z_int = torch.round(pre_act / pre_act_scaling_factor) 
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)

            m, e = batch_frexp(new_scale)

            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round( output / (2.0**e) )

            if identity is not None:
                # needs addition of identity activation
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                output = output1 + output

            if bit_num in [4, 8, 16]:
                if quant_mode == 'symmetric':
                    return torch.clamp( output.type(torch.float), -n - 1, n)
                else:
                    return torch.clamp( output.type(torch.float), 0, n)
            else:
                return output.type(torch.float)


    @ staticmethod
    def backward(ctx, grad_output):
        identity_grad = None
        if ctx.identity is not None:
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None,\
                identity_grad, None
