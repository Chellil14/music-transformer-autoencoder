from layers import transformer

from enum import Enum
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# TODO move to constants
NUM_NOTES = 88 # [21, 108]
NUM_VELOCITY_BINS = 32
NUM_TOKENS_PERF = NUM_NOTES * 2 + NUM_VELOCITY_BINS + 2 + 100 # PAD/EOS + pitch*2(on/off) + velocity + timeshift
NUM_TOKENS_MEL = NUM_NOTES + 2 + 2 # PAD/EOS + pitch + on/off

class MelodyCombineMethod(Enum):
    NONE = 0
    SUM = 1
    CONCAT = 2
    TILE = 3

class MusicAETransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        max_att_len: int,
        hidden_dim: int,
        filter_dim: int,
        dropout_rate: float,
        input_bias: bool,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        normalize_eps: float = 1e-9,
        melody_combine_method: str = 'NONE'
    ):
        super(MusicAETransformer, self).__init__()
        # TODO all transformer args + opts(melody, aggregation, ...)
        self.melody_combine_method = MelodyCombineMethod[melody_combine_method]
        self.has_melody = (self.melody_combine_method != MelodyCombineMethod.NONE)
        self.embedding_perf = nn.Embedding(NUM_TOKENS_PERF, hidden_dim)
        encoder_layer = transformer.EncoderLayer(
            num_heads,
            max_att_len,
            hidden_dim,
            filter_dim,
            dropout_rate,
            input_bias,
            False,
            qk_dim,
            v_dim,
            normalize_eps
        )
        # TODO layer norm?
        self.encoder_perf = transformer.TransformerEncoder(encoder_layer, num_layers)
        if self.has_melody:
            self.embedding_mel = nn.Embedding(NUM_TOKENS_MEL, hidden_dim)
            encoder_layer_melody = transformer.EncoderLayer(
                num_heads,
                max_att_len,
                hidden_dim,
                filter_dim,
                dropout_rate,
                input_bias,
                qk_dim,
                v_dim,
                normalize_eps
            )
            # TODO layer norm?
            self.encoder_melody = transformer.TransformerEncoder(encoder_layer, num_layers)
        else:
            self.encoder_melody = None
        decoder_layer = transformer.DecoderLayer(
            num_heads,
            max_att_len,
            hidden_dim,
            filter_dim,
            dropout_rate,
            input_bias,
            qk_dim,
            v_dim,
            normalize_eps
        )
        # TODO layer norm?
        self.decoder = transformer.TransformerDecoder(decoder_layer, num_layers)
        self.fc_softmax = nn.Sequential(
            nn.Linear(hidden_dim, NUM_TOKENS_PERF),
            nn.Softmax(-1)
        )

    def _combine(self, perf_enc, mel_enc):
        if self.melody_combine_method == MelodyCombineMethod.NONE:
            assert False, "_combine shouldn't be called if melody is not used"
        elif self.melody_combine_method == MelodyCombineMethod.SUM:
            return perf_enc + mel_enc
        elif self.melody_combine_method == MelodyCombineMethod.CONCAT:
            # FIXME correct bias should be used for the pad token
            # NOTE this assumes batch size of 1 or no padding on melody
            pad_token = torch.zeros_like(perf_enc)
            return torch.cat((mel_enc, pad_token, perf_enc), 1)
        elif self.melody_combine_method == MelodyCombineMethod.TILE:
            return torch.cat((mel_enc, perf_enc.expand(-1, mel_enc.shape[1], -1)), 2)
        else:
            assert False, f"unknown melody combine method {melody_combine_method}"

    def encode(self, performance, melody=None):
        perf_embedding = self.embedding_perf(performance)
        perf_enc = self.encoder_perf(perf_embedding)
        outputs = torch.mean(perf_enc, 1, keepdim=True) # mean aggregation
        if self.has_melody:
            assert melody is not None
            mel_embedding = self.embedding_perf(melody)
            mel_enc = self.encoder_mel(mel_embedding)
            outputs = self._combine(perf_enc, mel_enc)
        else:
            assert melody is None
        return outputs

    def decode(self, inputs, encoder_outputs):
        decoder_out = self.decoder(inputs, encoder_outputs)
        return self.fc_softmax(decoder_out)

    def shift_embed(self, performance):
        performance_shifted = F.pad(performance, (1, 0))[:, :-1]
        return self.embedding_perf(performance_shifted)

    def forward(self, performance, melody=None):
        encoder_outputs = self.encode(performance, melody)
        return self.decode(self.shift_embed(performance), encoder_outputs)
