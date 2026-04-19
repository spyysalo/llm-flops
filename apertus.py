#!/usr/bin/env python3

# Estimate compute cost for a dense GPT-like Hugging Face
# AutoModelForCausalLM based on its configuration using the
# implementation from the Apertus paper
# (https://arxiv.org/pdf/2509.14233#page=84)

import logging

from common import (
    get_head_dim,
    get_num_key_value_heads,
    get_intermediate_size,
    has_glu,
    set_up_logging,
    parse_args,
    set_up_config,
)


def attention_gqa_flops(
        seq_len: int,
        d_model: int,
        key_size: int,
        num_heads: int,
        num_kv_heads: int,
) -> int:
    assert num_heads % num_kv_heads == 0
    heads_per_kv = num_heads // num_kv_heads    # also unused in original

    q_proj = 2 * seq_len * d_model * (num_heads * key_size)
    k_proj = 2 * seq_len * d_model * (num_kv_heads * key_size)
    v_proj = k_proj
    qk = 2 * num_heads * seq_len * seq_len * key_size
    qk_norm = qk_norm_flops(seq_len, key_size, num_heads, num_kv_heads)
    softmax = 3 * num_heads * seq_len * seq_len
    attn_v = 2 * num_heads * seq_len * seq_len * key_size
    out_proj = 2 * seq_len * (num_heads * key_size) * d_model

    return (
        q_proj
        + k_proj
        + v_proj
        + qk
        + qk_norm
        + softmax
        + attn_v
        + out_proj
    )


def dense_mlp(seq_len, d_model, ffw_size, swiglu=False):
    if not swiglu:
        return 2 * seq_len * (2 * d_model * ffw_size)
    else:
        return 2 * seq_len * (3 * d_model * ffw_size)


def qk_norm_flops(
        seq_len: int, key_size: int, num_heads: int, num_kv_heads: int
) -> int:
    vectors = seq_len * (num_heads + num_kv_heads)
    return 4 * vectors * key_size


def rmsnorm(seq_len, d_model):
    return 4 * seq_len * d_model


def final_logits(seq_len, d_model, vocab_size):
    return 2 * seq_len * d_model * vocab_size


def get_flops(
        n_layers,
        seq_len,
        vocab_size,
        d_model,
        key_size,
        num_heads,
        num_kv_heads,
        ffw_size,
        swiglu=False,
):
    return (
        n_layers
        * (
            attention_gqa_flops(seq_len, d_model, key_size, num_heads, num_kv_heads)
            + dense_mlp(seq_len, d_model, ffw_size, swiglu=swiglu)
            + 2 * rmsnorm(seq_len, d_model)
        )
        + final_logits(seq_len, d_model, vocab_size)
    )


def apertus_estimate(config):
    """Estimate forward FLOPs per token using the implementation from
    the Apertus paper (https://arxiv.org/pdf/2509.14233#page=84).
    """
    total_flops = get_flops(
        n_layers=config.num_hidden_layers,
        seq_len=config.max_position_embeddings,
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        key_size=get_head_dim(config),
        num_heads=config.num_attention_heads,
        num_kv_heads=get_num_key_value_heads(config),
        ffw_size=get_intermediate_size(config),
        swiglu=has_glu(config),
    )
    tokens = config.max_position_embeddings
    return total_flops // tokens


def main():
    args = parse_args()
    set_up_logging(args)
    config = set_up_config(args)

    flops = apertus_estimate(config)
    print(f'Per token forward FLOPs estimates')
    print(f'apertus\t{flops}')


if __name__ == '__main__':
    main()
