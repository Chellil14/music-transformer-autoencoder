import einops
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

def mask_to_bias(mask: torch.Tensor) -> torch.Tensor:
    return 1e9 * (mask - 1)


def relative_logits(q: torch.Tensor, rel_k: torch.Tensor) -> torch.Tensor:
    """
    q.shape = (B, H, L, D)
    rel_k.shape = (2L-1, D)
    rel_logits.shape = (B, H, L, L)
    """
    b, h, l, d = q.shape
    target = {'device': q.device, 'dtype': q.dtype}
    if (*rel_k.shape,) == (2 * l - 1, d):
        # Shared key for all heads
        abs_logits = torch.einsum('bhld,md->bhlm', q, rel_k)
        # Each head has its own key
    elif (*rel_k.shape,) == (h, 2 * l - 1, d):
        abs_logits = torch.einsum('bhld,hmd->bhlm', q, rel_k)
    else:
        raise ValueError(f"The dimension of rel_k should be either "
                         f"({2 * l - 1}, {d}) or ({h}, {2 * l - 1}, {d}), "
                         f"but {(*rel_k.shape,)} is given")
    # get the relative logits by pad -> reshape -> slice to skew the tensor
    logits_pad1 = torch.cat((abs_logits, torch.zeros((b, h, l, 1), **target)), dim = 3)
    logits_flat = einops.rearrange(logits_pad1, 'b h l c-> b h (l c)')
    logits_pad2 = torch.cat((logits_flat, torch.zeros((b, h, l - 1), **target)), dim = 2)
    rel_logits = logits_pad2.reshape(b, h, l + 1, 2 * l - 1)[:, :, :l, (l-1):]
    return rel_logits


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout_rate: float,
        input_bias: bool,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None
    ):
        """
        num_heads: The number of heads
        max_att_len: Maximum distance for relative positional encoding
        dropout_rate:
        hidden_dim: Hidden dimension for input/output
        qk_dim: (optional) Hidden dimension for transformed query and key
        v_dim: (optional) Hidden dimension for transformed query
        """
        # TODO add direction
        # TODO also init Wq Wk Wv(q: rsqrt(hidden_dim * (key_dim // num_heads)), k/v: rsqrt(hidden_dim))
        super(MultiHeadAttention, self).__init__()

        if qk_dim is None:
            qk_dim = hidden_dim
        if v_dim is None:
            v_dim = hidden_dim
        assert qk_dim % num_heads == 0
        assert v_dim % num_heads == 0

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dim_per_head = hidden_dim // num_heads

        # TODO init properly
        self.Wq = nn.Linear(hidden_dim, qk_dim, bias=input_bias)
        self.Wk = nn.Linear(hidden_dim, qk_dim, bias=input_bias)
        self.Wv = nn.Linear(hidden_dim, v_dim, bias=input_bias)
        self.fc = nn.Linear(v_dim, hidden_dim) # TODO check if this needs bias
        self.dropout = nn.Dropout(dropout_rate) # NOTE before mutliplying to v

        nn.init.normal_(self.Wq.weight, std=((hidden_dim * (qk_dim // num_heads)) ** -0.5))
        nn.init.normal_(self.Wk.weight, std=(hidden_dim ** -0.5))
        nn.init.normal_(self.Wv.weight, std=(hidden_dim ** -0.5))
        nn.init.normal_(self.fc.weight, std=(v_dim ** -0.5))

    def _get_transformed_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None) -> Tuple[torch.Tensor, ...]:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        v_len = v.shape[-2]
        assert k_len == v_len

        q = einops.rearrange(self.Wq(q), 'b l (h d) -> b h l d', h=self.num_heads)
        k = einops.rearrange(self.Wk(k), 'b l (h d) -> b h l d', h=self.num_heads)
        v = einops.rearrange(self.Wv(v), 'b l (h d) -> b h l d', h=self.num_heads)

        return q, k, v

    def _get_qk_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # return q @ k.mT # requires torch>=1.11
        return q @ k.transpose(-2, -1)

    def _calc_attention_weight(self, product: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_factor = self.dim_per_head ** -0.5
        _, _, q_len, k_len = product.shape
        target = {'device': product.device, 'dtype': product.dtype}

        mask_bias = torch.zeros(q_len, k_len, **target)
        if mask is not None:
            assert (*mask.shape,) == (q_len, k_len)
            mask_bias = mask_to_bias(mask)

        # TODO consider using masked_fill_ for invisible pairs
        return self.dropout(F.softmax(product * scale_factor + mask_bias, dim=-1))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        (B, L, D), (B, L, D), (B, L, D) -> (B, L, D)
        mask: 
        """
        q, k, v = self._get_transformed_qkv(q, k, v)
        attention_weight = self._calc_attention_weight(self._get_qk_product(q, k), mask)
        attention_product = einops.rearrange(attention_weight @ v, 'b h l d -> b l (h d)')
        return self.fc(attention_product)


class RelativeGlobalAttention(MultiHeadAttention):
    def __init__(
        self,
        num_heads: int,
        max_att_len: int,
        hidden_dim: int,
        dropout_rate: float,
        input_bias: bool,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None
    ):
        """
        num_heads: The number of heads
        max_att_len: Maximum distance for relative positional encoding
        dropout_rate:
        hidden_dim: Hidden dimension for input/output
        qk_dim: (optional) Hidden dimension for transformed query and key
        v_dim: (optional) Hidden dimension for transformed query
        """
        super(RelativeGlobalAttention, self).__init__(
            num_heads,
            hidden_dim,
            dropout_rate,
            input_bias,
            qk_dim,
            v_dim
        )

        self.max_att_len = max_att_len

        # TODO init properly, change size in unidirectional case?
        self.relative_key = nn.Parameter(torch.randn(self.num_heads, self.max_att_len * 2 - 1, self.qk_dim // self.num_heads))

    def _get_rel_k(self, q_len: int) -> torch.Tensor:
        pad_len = max(q_len - self.max_att_len, 0)
        slice_begin = max(self.max_att_len - q_len, 0)
        slice_end = slice_begin + q_len * 2 - 1
        return F.pad(self.relative_key, (0, 0, pad_len, pad_len))[:, slice_begin:slice_end, :]

    def _get_qk_product(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # return q @ k.mT + relative_logits(q, self._get_rel_k()) # requires torch>=1.11
        return q @ k.transpose(-2, -1) + relative_logits(q, self._get_rel_k(q.shape[-2]))
