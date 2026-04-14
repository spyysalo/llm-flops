#!/usr/bin/env python3

from kaplan_et_al import two_n_estimate, table_1_estimate
from fvcore_flops import fvcore_2x_estimate, fvcore_custom_op_estimate
from meta_flops import meta_estimate

from common import (
    set_up_logging,
    parse_args,
    set_up_config,
)


two_n_nohead = lambda c: two_n_estimate(c, exclude_lm_head=True)
two_n_head = lambda c: two_n_estimate(c, exclude_lm_head=False)
table_1_no_emb = lambda c: table_1_estimate(c, exclude_embeddings=True)
table_1_emb = lambda c: table_1_estimate(c, exclude_embeddings=False)


estimates = {
    'two_n_nohead': two_n_nohead,
    'two_n_head': two_n_head,
    'table_1_no_emb': table_1_no_emb,
    'table_1_emb': table_1_emb,
    'meta': meta_estimate,
    'fvcore_custom_op': fvcore_custom_op_estimate,
    'fvcore_2x': fvcore_2x_estimate,
}


def main():
    args = parse_args()
    set_up_logging(args)
    config = set_up_config(args)

    flops = { n: e(config) for n, e in estimates.items() }
    flops_max = max(flops.values())

    print(f'Per token forward FLOPs estimates (% of max)')
    for n, f in flops.items():
        print(f'{n}\t{f}\t({f/flops_max:.2%})')


if __name__ == '__main__':
    main()
