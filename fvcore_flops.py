#!/usr/bin/env python3

# Estimate compute cost for a Hugging Face AutoModelForCausalLM using
# fvcore. Implements two approaches to roughly accommodate for fvcore
# counting each fused multiply-add as one FLOP
# (https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis): doubling the total, and using custom ops that primarily
# double the count for selected operations (linear, matmul, etc.).

import re
import math
import logging

import torch
import torch.nn as nn

from collections import Counter
from transformers import AutoModelForCausalLM

try:
    from transformers import PreTrainedConfig
except ImportError:
    from transformers import PretrainedConfig as PreTrainedConfig

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import (
    get_shape,
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


# ----------------------------------------------------------------------------
# Custom fvcore ops
# ----------------------------------------------------------------------------


def ignore(inputs, outputs):
    return 0


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


def softmax_flops(inputs, outputs):
    """Estimate FLOPs for aten::softmax."""
    input_shape = get_shape(inputs[0])
    return 3 * math.prod(input_shape)


def num_output_elements(inputs, outputs):
    """Return the number of output elements as FLOPs estimate."""
    assert len(outputs) == 1
    output_shape = get_shape(outputs[0])
    return math.prod(output_shape)


def print_shapes(inputs, outputs):
    # debugging support
    input_shapes = [get_shape(i) for i in inputs]
    for i, s in enumerate(input_shapes):
        print('INPUT', i, s)
    output_shapes = [get_shape(i) for i in outputs]
    for i, s in enumerate(output_shapes):
        print('OUTPUT', i, s)
    return 0


CUSTOM_OPS = {
    "aten::embedding": ignore,
    "aten::addmm": addmm_2flop_fma,
    "aten::linear": linear_2flop_fma,
    "aten::matmul": matmul_2flop_fma,
    "aten::bmm": bmm_2flop_fma,
    "aten::softmax": softmax_flops,
    "aten::add": num_output_elements,
    "aten::mul": num_output_elements,
    "aten::div": num_output_elements,
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
    ).set_op_handle(**CUSTOM_OPS)
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
