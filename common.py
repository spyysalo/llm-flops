#!/usr/bin/env python3

# Support functions common to various compute cost estimates.

import torch

import logging

from argparse import ArgumentParser

from transformers import AutoConfig, AutoModelForCausalLM

try:
    from transformers import PreTrainedConfig
except ImportError:
    from transformers import PretrainedConfig as PreTrainedConfig

# A high seq len value (e.g. for a long-context model) can lead to an
# exaggerated estimate of attention compute cost compared to actual
# pretraining cost. Warn for seq lens above this.
SEQ_LEN_WARN_LIMIT = 10000


# ----------------------------------------------------------------------------
# Functions for parameters that are not (always) explicit in configuration
# ----------------------------------------------------------------------------

def has_tied_embeddings(config: PreTrainedConfig) -> bool:
    """Return True if model has tied embeddings, False otherwise."""
    try:
        return config.tie_word_embeddings
    except AttributeError:
        pass
    raise ValueError(f'cannot determine tied embeddings for model_type {config.model_type}')


def has_trainable_positional_embeddings(config: PreTrainedConfig) -> bool:
    """Return True if model has trainable positional embeddings, False
    otherwise."""
    if config.model_type == 'gpt2':
        return True

    rope_params = [p for p in config if 'rope' in p]
    if rope_params:
        logging.warning(f'rope parameter(s) in config ({rope_params}), assuming no trainable positional embeddings')
        return False

    raise ValueError(f'cannot determine trainable positional embeddings for model_type {config.model_type}')


def get_num_key_value_heads(config: PreTrainedConfig) -> int:
    """Returns the number of key-value heads in the model."""
    try:
        return config.num_key_value_heads
    except AttributeError:
        pass
    logging.warning(f'num_key_value_heads not in config, using num_attention_heads ({config.num_attention_heads})')
    return config.num_attention_heads


def get_attention_bias(config: PreTrainedConfig) -> bool:
    """Returns True if attention projections have bias, False otherwise."""
    try:
        return config.attention_bias
    except AttributeError:
        pass
    if config.model_type == 'gpt2':
        return True
    else:
        raise ValueError(f'cannot determine attention bias for model_type {config.model_type}')


def get_mlp_bias(config: PreTrainedConfig) -> bool:
    """Returns True if MLP projections have bias, False otherwise."""
    # config.mlp_bias is not consistently present in config, so list
    # models by whether they use it
    MLP_BIASED_MODELS = {
        'gpt2',
        'gpt_neox',
    }
    MLP_UNBIASED_MODELS = {
        'qwen3',
    }
    try:
        return config.mlp_bias
    except AttributeError:
        pass
    if config.model_type in MLP_BIASED_MODELS:
        return True
    elif config.model_type in MLP_UNBIASED_MODELS:
        return False
    else:
        raise ValueError(f'cannot determine mlp bias for model_type {config.model_type}')


def has_glu(config: PreTrainedConfig) -> bool:
    """Return True if model uses GLU in MLPs, False otherwise."""
    # AutoConfig does not offer a direct way to look up whether the
    # model uses GLU in MLPs, so list by model_type instead.
    GLU_MODEL_TYPES = {
        'llama',
        'qwen3',
    }
    GLULESS_MODEL_TYPES = {
        'gpt2',
        'gpt_neox',
    }
    if config.model_type in GLU_MODEL_TYPES:
        return True
    elif config.model_type in GLULESS_MODEL_TYPES:
        return False
    else:
        raise ValueError(f'cannot determine whether model_type {config.model_type} uses GLU')


def get_head_dim(config: PreTrainedConfig) -> int:
    """Returns the attention head dimension of a model configuration."""
    try:
        return config.head_dim
    except AttributeError:
        pass

    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(f'head_dim not in config and hidden_size {config.hidden_size} % num_attention_heads {config.num_attention_heads} != 0')

    head_dim = config.hidden_size // config.num_attention_heads
    if config.model_type != 'gpt2':    # known to hold for gpt2
        logging.warning(f'head_dim not in config, assuming hidden/heads ({head_dim})')
    return head_dim


def get_intermediate_size(config: PreTrainedConfig) -> int:
    """Returns the MLP intermediate size of a model configuration."""
    try:
        return config.intermediate_size
    except AttributeError:
        pass
    if config.model_type == 'gpt2':
        return 4 * config.hidden_size
    else:
        raise ValueError(f'cannot determine intermediate size for model_type {config.model_type}')


def get_num_parameters(config: PreTrainedConfig,
                       exclude_input_embedding: bool = True,
                       exclude_lm_head: bool = False):
    """Estimates the number of parameters of a model configuration.

    Args:
        config: PreTrainedConfig for the model
        exclude_input_embedding: If True, excludes input embedding
        exclude_lm_head: If True, excludes output unembedding (LM head)

    Returns:
        int: Estimated number of parameters

    When input embedding and LM head weights are tied, their
    parameters are counted only once and only excluded if both
    `exclude_input_embedding` and `exclude_lm_head` are True.
    """
    with torch.device('meta'):    # shape only, no real memory use
        model = AutoModelForCausalLM.from_config(config)

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
        
    total_params = sum(p.numel() for p in model.parameters())
    input_params = sum(p.numel() for p in input_embedding.parameters())
    output_params = sum(p.numel() for p in output_embedding.parameters())

    tied = input_embedding.weight is output_embedding.weight

    logging.info(f'total_params {total_params}')
    logging.info(f'input_params {input_params}')
    logging.info(f'output_params {output_params}')
    logging.info(f'tied {tied}')
    
    # sanity checks
    assert tied == config.tie_word_embeddings
    assert total_params == model.num_parameters()
    
    count = total_params
    if tied:
        if exclude_input_embedding and exclude_lm_head:
            count -= input_params
    else:
        if exclude_input_embedding:
            count -= input_params
        if exclude_lm_head:
            count -= output_params

    return count


# ----------------------------------------------------------------------------
# CLI and misc.
# ----------------------------------------------------------------------------

class DuplicateFilter(logging.Filter):
    """Filter out duplicate logging messages"""
    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        msg = record.getMessage()
        if msg in self.seen:
            return False
        self.seen.add(msg)
        return True


def set_up_logging(args=None):
    if args is not None and args.verbose:
        level = logging.INFO
    elif args is not None and args.quiet:
        level = logging.ERROR
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    logging.getLogger().addFilter(DuplicateFilter())


def create_dummy_input(config, batch_size=1):
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.max_position_embeddings),
    )


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('config', help='HF config name or path')
    ap.add_argument('--seq-len', type=int,
                    help='override max_position_embeddings')    
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    return ap.parse_args()


def set_up_config(args):
    config = AutoConfig.from_pretrained(args.config)

    if args.seq_len is not None:
        config.max_position_embeddings = args.seq_len
    elif config.max_position_embeddings > SEQ_LEN_WARN_LIMIT:
        logging.warning(
            f'high config.max_position_embeddings value '
            f'({config.max_position_embeddings}), consider using '
            f'--seq-len argument'
        )

    return config
