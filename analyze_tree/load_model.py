import torch
from torch import nn
from torch.nn import functional as F

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
import model


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model, criterion, optimizer = torch.load(f, map_location='cpu')

    return model
