import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from layers.transformer_layers import EncoderLayer, DecoderLayer

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm_layer = None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm_layer = norm_layer

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs) # TODO add more args if needed

        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        return outputs


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm_layer = None):
        super(TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm_layer = norm_layer

    def forward(self, inputs, enc_outputs = None, enc_mask = None):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, enc_outputs, enc_mask) # TODO add more args if needed

        if self.norm_layer is not None:
            outputs = self.norm_layer(outputs)

        return outputs
