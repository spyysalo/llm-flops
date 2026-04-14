#!/usr/bin/env python3

# Estimate compute cost for a dense GPT-like Hugging Face
# AutoModelForCausalLM based on its configuration using estimates
# suggested by Kaplan et al. (2020) "Scaling Laws for Neural Language
# Models" (https://arxiv.org/pdf/2001.08361).

import logging

from argparse import ArgumentParser

from transformers import AutoModelForCausalLM, PreTrainedConfig

from common import (
    get_num_parameters,
    get_head_dim,
    get_intermediate_size,
    set_up_logging,
    parse_args,
    set_up_config,
)


def two_n_estimate(config: PreTrainedConfig,
                   exclude_lm_head: bool = False) -> int:
    """Estimate forward FLOPs per token using the simple 2N
    approximation.
    """
    N = get_num_parameters(config, exclude_lm_head=exclude_lm_head)
    return 2 * N


def table_1_estimate(config: PreTrainedConfig,
                     exclude_embeddings: bool = False) -> int:
    """Estimate forward FLOPs per token using the equations from
    Table 1 in Kaplan et al. (2020) "Scaling Laws for Neural Language
    Models" (https://arxiv.org/pdf/2001.08361#page=7).
    """
    # rename for consistency with the paper
    n_vocab = config.vocab_size
    n_ctx = config.max_position_embeddings
    n_layer = config.num_hidden_layers
    d_model = config.hidden_size
    d_embd = d_model
    d_attn = config.num_attention_heads * get_head_dim(config)
    d_ff = get_intermediate_size(config)

    # "FLOPs per Token" column
    embed = 4 * d_model
    attn_qkv = 2 * n_layer * d_model * 3 * d_attn
    attn_mask = 2 * n_layer * n_ctx * d_attn
    attn_proj = 2 * n_layer * d_attn * d_embd
    ffwd = 2 * n_layer * 2 * d_model * d_ff
    deembed = 2 * d_model * n_vocab

    logging.info(f'embed FLOPs per token {embed}')
    logging.info(f'attn_qkv FLOPs per token {attn_qkv}')
    logging.info(f'attn_mask FLOPs per token {attn_mask}')
    logging.info(f'attn_proj FLOPs per token {attn_proj}')
    logging.info(f'ffwd FLOPs per token {ffwd}')
    logging.info(f'deembed FLOPs per token {deembed}')

    N = 2 * d_model * n_layer * (2 * d_attn + d_ff)
    C_forward = 2 * N + 2 * n_layer * n_ctx * d_attn

    # sanity check
    assert C_forward == attn_qkv + attn_mask + attn_proj + ffwd

    if exclude_embeddings:
        return C_forward
    else:
        return embed + C_forward + deembed


def main():
    args = parse_args()
    set_up_logging(args)
    config = set_up_config(args)

    two_n_nohead = two_n_estimate(config, exclude_lm_head=True)
    two_n_head = two_n_estimate(config, exclude_lm_head=False)
    t1_noemb = table_1_estimate(config, exclude_embeddings=True)
    t1_emb = table_1_estimate(config, exclude_embeddings=False)

    print(f'Per token forward FLOPs estimates')
    print(f'2N, no LM head\t{two_n_nohead}')
    print(f'Table 1, no embeddings\t{t1_noemb}')
    print(f'2N, LM head\t{two_n_head}')
    print(f'Table 1, embeddings\t{t1_emb}')


if __name__ == '__main__':
    main()
