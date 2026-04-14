#!/usr/bin/env python3

# Estimate compute cost for a Hugging Face AutoModelForCausalLM using
# fvcore. Implements two approaches to roughly accommodate for fvcore
# counting each fused multiply-add as one FLOP
# (https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis): doubling the total, and using custom ops that double
# the count for selected operations (linear, matmul, etc.).

import re
import logging

import torch
import torch.nn as nn

from collections import Counter
from transformers import AutoModelForCausalLM, PreTrainedConfig

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import (
    addmm_flop_jit,
    bmm_flop_jit,
    linear_flop_jit,
    matmul_flop_jit,
)

from common import (
    set_up_logging,
    parse_args,
    set_up_config,
    create_dummy_input,
)


class Wrapper(nn.Module):
    """Minimal HF model wrapper for fvcore FlopCountAnalysis."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits


def addmm_2flop_fma(inputs, outputs):
    """Approximate counting each FMA as two FLOPs in addmm."""
    return 2 * addmm_flop_jit(inputs, outputs)


def bmm_2flop_fma(inputs, outputs):
    """Approximate counting each FMA as two FLOPs in bmm."""
    return 2 * bmm_flop_jit(inputs, outputs)


def linear_2flop_fma(inputs, outputs):
    """Approximate counting each FMA as two FLOPs in linear."""
    return 2 * linear_flop_jit(inputs, outputs)


def matmul_2flop_fma(inputs, outputs):
    """Approximate counting each FMA as two FLOPs in matmul."""
    return 2 * matmul_flop_jit(inputs, outputs)


CUSTOM_2FLOP_FMA_OPS = {
    "aten::addmm": addmm_2flop_fma,
    "aten::linear": linear_2flop_fma,
    "aten::matmul": matmul_2flop_fma,
    "aten::bmm": bmm_2flop_fma,
}


def create_model(config):
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation='eager'
    )
    model.eval()
    wrapped_model = Wrapper(model)
    return wrapped_model


def log_by_module(flops):
    grouped = Counter()
    for k, v in flops.by_module().items():
        k = re.sub(r'\.\d+\.', '.<layer>.', k)
        k = re.sub(r'\.\d+$', '.<layer>', k)
        grouped[k] += v
    total = flops.total()
    for k, v in grouped.items():
        logging.info(f'{k}: {v} ({v/total:.1%})')


def fvcore_2x_estimate(config):
    """Estimate forward FLOPs per token as 2x the fvcore estimate to
    accommodate for fvcore counting each fused multiply-add as one
    FLOP instead of two.
    """
    input_ids = create_dummy_input(config, batch_size=1)
    flops = FlopCountAnalysis(
        create_model(config),
        input_ids
    )
    log_by_module(flops)
    tokens = input_ids.numel()
    return 2 * flops.total() // tokens


def fvcore_custom_op_estimate(config):
    """Estimate forward FLOPs per token as using custom ops that
    approximate counting each FMA as two FLOPs.
    """
    input_ids = create_dummy_input(config, batch_size=1)
    flops = FlopCountAnalysis(
        create_model(config),
        input_ids
    ).set_op_handle(**CUSTOM_2FLOP_FMA_OPS)
    log_by_module(flops)
    tokens = input_ids.numel()
    return flops.total() // tokens


def main():
    args = parse_args()
    set_up_logging(args)
    config = set_up_config(args)

    fvcore_2x = fvcore_2x_estimate(config)
    fvcore_custom_ops = fvcore_custom_op_estimate(config)

    print(f'Per token forward FLOPs estimates')
    print(f'2x fvcore\t{fvcore_2x}')
    print(f'fvcore custom ops\t{fvcore_custom_ops}')


if __name__ == '__main__':
    main()
