import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torchvision.ops import DeformConv2d

#TODO:  Proposed Deformable Convolutional Residual Attention Module
class InvertedResidual(nn.Module):
        ""
        The program will be uploaded as soon as the article is published
        ""
        return ()
#TODO: KAN Module
class KANLinear(torch.nn.Module):
        ""
        The program will be uploaded as soon as the article is published
        ""
        return ()

#TODO: Positional coding: making the network capable of temporal learning
class FixedPositionalEncoding(nn.Module):
        ""
        The program will be uploaded as soon as the article is published
        ""
        return x.transpose(0, 1)


#TODO: DCRA enhanced KAN model
class DCNNKAN(nn.Module):
        ""
        The program will be uploaded as soon as the article is published
        ""
        return x

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}")
