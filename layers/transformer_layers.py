import einops
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from layers.attention import MultiHeadAttention, RelativeGlobalAttention

def get_attention_mask(seq_len: int, backward_len: int, forward_len: int) -> torch.Tensor:
    """
    Get mask for attention, 1 for allowed region
    """
    return torch.ones(seq_len, seq_len).triu(-backward_len).tril(forward_len)

def get_unidirectional_mask(seq_len: int, max_att_len: Optional[int] = None) -> torch.Tensor:
    if max_att_len is None:
        max_att_len = seq_len
    return get_attention_mask(seq_len, max_att_len, 0)


def get_bidirectional_mask(seq_len: int, max_att_len: Optional[int] = None) -> torch.Tensor:
    if max_att_len is None:
        max_att_len = seq_len
    return get_attention_mask(seq_len, max_att_len, max_att_len)


class TwoLayerFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float):
        super(TwoLayerFFN, self).__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU() # TODO allow changing this

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.dropout(self.activation(self.fc0(inputs))))


class PrePostProcLayer(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        dropout_rate: float,
        normalized_shape: List[int],
        normalize_eps: float,
        normalize_before: bool = False
    ):
        """
        module: The module to be wrapped
        dropout_rate: The rate for dropout layer
        normalized_shape: List of trailing dimensions to be normalized(usually the last one)
        normalize_before: If false, ("", "dan") as in the original transformer paper.
                          If true, ("n", "da") like transformer_base_v2 in tensor2tensor.
        """
        super(PrePostProcLayer, self).__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape, normalize_eps)
        self.normalize_before = normalize_before

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._postprocess(self.module(self._preprocess(x), *args, **kwargs), x)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_before:
            x = self.layer_norm(x)
        return x

    def _postprocess(self, x: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x += x_orig
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        max_att_len: int,
        hidden_dim: int,
        filter_dim: int,
        dropout_rate: float,
        input_bias: bool,
        bidirectional: bool,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        normalize_eps: float = 1e-9
    ):
        super(EncoderLayer, self).__init__()

        self_attn = RelativeGlobalAttention(
            num_heads,
            max_att_len,
            hidden_dim,
            dropout_rate,
            input_bias,
            qk_dim,
            v_dim
        )
        self.self_attn_wrapped = PrePostProcLayer(self_attn, dropout_rate, [hidden_dim], normalize_eps, False)

        ffn = TwoLayerFFN(hidden_dim, filter_dim, hidden_dim, dropout_rate)
        self.ffn_wrapped = PrePostProcLayer(ffn, dropout_rate, [hidden_dim], normalize_eps, False)

        self.bidirectional = bidirectional

    def forward(self, inputs: torch.Tensor):
        # TODO mask from sequence length
        B, L, D = inputs.shape
        target = {'device': inputs.device, 'dtype': inputs.dtype}
        mask = None
        if not self.bidirectional:
            mask = get_unidirectional_mask(L).to(**target)
        self_attn_out = self.self_attn_wrapped(inputs, inputs, inputs, mask)
        ffn_out = self.ffn_wrapped(self_attn_out)
        return ffn_out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        max_att_len: int,
        hidden_dim: int,
        filter_dim: int,
        dropout_rate: float,
        input_bias: bool,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        normalize_eps: float = 1e-9
    ):
        super(DecoderLayer, self).__init__()

        self_attn = RelativeGlobalAttention(
            num_heads,
            max_att_len,
            hidden_dim,
            dropout_rate,
            input_bias,
            qk_dim,
            v_dim
        )
        self.self_attn_wrapped = PrePostProcLayer(self_attn, dropout_rate, [hidden_dim], normalize_eps, False)

        enc_dec_attn = MultiHeadAttention(
            num_heads,
            hidden_dim,
            dropout_rate,
            input_bias,
            qk_dim,
            v_dim
        )
        self.enc_dec_attn_wrapped = PrePostProcLayer(enc_dec_attn, dropout_rate, [hidden_dim], normalize_eps, False)

        ffn = TwoLayerFFN(hidden_dim, filter_dim, hidden_dim, dropout_rate)
        self.ffn_wrapped = PrePostProcLayer(ffn, dropout_rate, [hidden_dim], normalize_eps, False)

    def forward(self, inputs: torch.Tensor, enc_outputs: Optional[torch.Tensor] = None, enc_mask: Optional[torch.Tensor] = None):
        """
        inputs: [B, L, D], D = self.hidden_dim
        enc_outputs: (Optional) [B, L_enc, D], D = self.hidden_dim
        enc_mask: (Optional) [B, L_enc, L]??
        """
        B, L, D = inputs.shape
        target = {'device': inputs.device, 'dtype': inputs.dtype}
        mask = get_unidirectional_mask(L).to(**target)
        self_attn_out = self.self_attn_wrapped(inputs, inputs, inputs, mask)
        total_attn = self_attn_out
        if enc_outputs is not None:
            # TODO add enc-dec mask for padding tokens
            total_attn += self.enc_dec_attn_wrapped(inputs, enc_outputs, enc_outputs, enc_mask)
        ffn_out = self.ffn_wrapped(self_attn_out)
        return ffn_out
