import numpy as np

def compute_weight(loss, norm):
    """ Compute the weight based on the pinball loss of the forecasts
    args:
        loss: float, pinball loss
        norm: str, normalization method
    returns:
        weight: float, weight"""
    if norm=='sum':
        weight = 1/loss
    elif norm=='softmax':
        weight = 1/np.exp(loss)
    else:
        raise ValueError('Not a valid normalization method')
    return weight