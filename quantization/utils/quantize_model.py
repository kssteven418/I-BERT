from quantization.utils.quant_modules import *

def freeze_model(model):
    """
    freeze the activation range
    """
    #print(type(model))
    if type(model) == QuantAct:
        model.fix()
    elif type(model) == QuantLinear:
        model.fix()
    elif type(model) == QuantLayerNorm:
        model.fix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    elif type(model) == nn.ModuleList:
        for n in model:
            freeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                freeze_model(mod)

def unfreeze_model(model):
    """
    unfreeze the activation range
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == QuantLinear:
        model.unfix()
    elif type(model) == QuantLayerNorm:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    elif type(model) == nn.ModuleList:
        for n in model:
            unfreeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                unfreeze_model(mod)
