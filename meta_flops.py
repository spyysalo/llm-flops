#!/usr/bin/env python3

# Estimate compute cost for a dense GPT-like Hugging Face
# AutoModelForCausalLM based on its configuration using a somewhat
# generic model on the meta device.

import re
import math
import logging

import torch

from collections import Counter, defaultdict
from argparse import ArgumentParser

from torch import nn

from common import (
    has_tied_embeddings,
    has_trainable_positional_embeddings,
    get_num_key_value_heads,
    get_attention_bias,
    get_mlp_bias,
    has_glu,
    get_head_dim,
    get_intermediate_size,
    set_up_logging,
    parse_args,
    set_up_config,
    create_dummy_input,
)


# ----------------------------------------------------------------------------
# Model and other modules
# ----------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tied_embeddings = has_tied_embeddings(config)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size)

        if has_trainable_positional_embeddings(config):
            self.embed_positions = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        else:
            self.embed_positions = None

        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.lm_head = Linear(
            config.hidden_size, config.vocab_size, bias=False, name='lm_head')
        if self.tied_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, x):
        # TODO position embeddings
        x = self.embed_tokens(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return self.lm_head(x)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        # TODO determine if RMSNorm or something else
        self.norm_1 = RMSNorm(config.hidden_size)
        self.norm_2 = RMSNorm(config.hidden_size)

    def forward(self, x):
        # TODO: residuals
        x = self.norm_1(x)
        x = self.attn(x)
        x = self.norm_2(x)
        x = self.mlp(x)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = get_head_dim(config)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = get_num_key_value_heads(config)
        attention_bias = get_attention_bias(config)

        self.q_proj = Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            attention_bias,
            name='attn.q_proj',
        )
        self.k_proj = Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            attention_bias,
            name='attn.k_proj',
        )
        self.v_proj = Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            attention_bias,
            name='attn.v_proj',
        )
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            attention_bias,
            name='attn.o_proj',
        )
        # TODO determine if QKnorm used
        self.q_norm = RMSNorm(self.head_dim, 'attn.qknorm')
        self.k_norm = RMSNorm(self.head_dim, 'attn.qknorm')

    def forward(self, x):
        # In part following HF transformers modeling_qwen3.py
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(hidden_shape).transpose(1,2))
        k = self.k_norm(self.k_proj(x).view(hidden_shape).transpose(1,2))
        v = self.v_proj(x).view(hidden_shape).transpose(1,2)

        # TODO: possible positional embeddings
        if self.num_key_value_heads != self.num_attention_heads:
            num_key_value_groups = (
                self.num_attention_heads // self.num_key_value_heads)
            k = k.repeat_interleave(num_key_value_groups, dim=1)
            v = v.repeat_interleave(num_key_value_groups, dim=1)

        # TODO: scaling, masking, dropout
        attn_weights = matmul(q, k.transpose(2, 3), 'attn.qk')
        attn_weights = softmax(attn_weights, -1, 'attn.softmax')

        attn_output = matmul(attn_weights, v, 'attn.av')
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        intermediate_size = get_intermediate_size(config)
        mlp_bias = get_mlp_bias(config)

        self.up_proj = Linear(
            config.hidden_size, intermediate_size, mlp_bias, 'mlp.up_proj')
        if has_glu(config):
            self.gate_proj = Linear(
                config.hidden_size, intermediate_size, mlp_bias, 'mlp.gate_proj')
        else:
            self.gate_proj = None
        self.down_proj = Linear(
            intermediate_size, config.hidden_size, mlp_bias, 'mlp.down_proj')

    def forward(self, x):
        # TODO: activations
        p = self.up_proj(x)

        if self.gate_proj is not None:    # GLU
            g = self.gate_proj(x)
            p = p * g
            log_flops('mlp.glu_mul', g.numel())    # elementwise multiplication

        o = self.down_proj(p)
        return o


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, label='rmsnorm'):
        super().__init__()
        self.label = label
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = 1e-6    # TODO

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        log_flops(self.label, 4*x.numel())
        return self.weight * x


# ----------------------------------------------------------------------------
# FLOP estimation and logging
# ----------------------------------------------------------------------------

def log_flops(label, count):
    log_flops.count[label] += count
log_flops.count = Counter()


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, name=None):
        super().__init__(in_features, out_features, bias=bias)
        self.name = name

    def forward(self, x):
        log_flops(self.name, linear_flops(x, self))
        return super().forward(x)


def matmul(a, b, name=None):
    log_flops(name, matmul_flops(a, b))
    return torch.matmul(a, b)


def softmax(input, dim, name=None):
    log_flops(name, 3*input.numel())
    return nn.functional.softmax(input, dim)


def linear_flops(x: torch.Tensor, layer: nn.Linear) -> int:
    """Estimates forward FLOPs for Linear layer applied to input Tensor x.

    Multiplications and additions are counted separately (one FLOP each).
    If the layer has a bias term, additions are included.

    Args:
        x: Tensor with shape [..., in_features]
        layer: nn.Linear layer with in_features and out_features

    Returns:
        int: Estimated number of FLOPs
    """
    assert x.shape[-1] == layer.in_features
    n = math.prod(x.shape[:-1])
    flops = 2 * n * layer.in_features * layer.out_features
    if layer.bias is not None:
        flops += n * layer.out_features
    return flops


def matmul_flops(a: torch.Tensor, b: torch.Tensor) -> int:
    """Estimates FLOPs for torch.matmul(a, b) where both are rank >= 2.

    Args:
        a: Tensor with shape [..., n, m]
        b: Tensor with shape [..., m, p]

    Returns:
        int: Estimated number of FLOPs
    """
    assert a.shape[-1] == b.shape[-2]
    n, m, p = a.shape[-2], a.shape[-1], b.shape[-1]
    broadcast_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    c = math.prod(broadcast_shape) if broadcast_shape else 1
    return 2 * c * m * n * p


def meta_estimate(config):
    with torch.device('meta'):
        model = Model(config)
        input_ids = create_dummy_input(config, batch_size=16)
    logits = model(input_ids)
    tokens = input_ids.numel()

    for label, count in log_flops.count.items():
        logging.info(f'{label} {count//tokens}')
    grouped = Counter()
    for label, count in log_flops.count.items():
        grouped[label.split('.')[0]] += count
    for group, count in grouped.items():
        logging.info(f'{group} {count//tokens}')

    total_flops = log_flops.count.total()
    return total_flops // tokens


def main():
    args = parse_args()
    set_up_logging(args)
    config = set_up_config(args)

    flops = meta_estimate(config)
    print(f'Per token forward FLOPs estimates')
    print(f'meta model\t{flops}')


if __name__ == '__main__':
    main()
