#!/usr/bin/env python3

from kaplan_et_al import two_n_estimate, table_1_estimate
from fvcore_flops import fvcore_2x_estimate, fvcore_custom_op_estimate
from meta_flops import meta_estimate
from apertus import apertus_estimate

from common import (
    set_up_logging,
    parse_args,
    set_up_config,
)


two_n_no_lm_head = lambda c: two_n_estimate(c, exclude_lm_head=True)
two_n_lm_head = lambda c: two_n_estimate(c, exclude_lm_head=False)
table_1_no_emb = lambda c: table_1_estimate(c, exclude_embeddings=True)
table_1_emb = lambda c: table_1_estimate(c, exclude_embeddings=False)


estimates = {
    '2N (no de-embed)': two_n_no_lm_head,
    '2N (w/de-embed)': two_n_lm_head,
    'Table 1 (no emb)': table_1_no_emb,
    'table 1 (w/emb)': table_1_emb,
    'meta': meta_estimate,
    'Apertus': apertus_estimate,
    '2x fvcore': fvcore_2x_estimate,
    'fvcore w/custom ops': fvcore_custom_op_estimate,
}


def main():
    args = parse_args()
    set_up_logging(args)
    config = set_up_config(args)

    flops = { n: e(config) for n, e in estimates.items() }
    flops_max = max(flops.values())

    print(f'Per token FW FLOPs\t(% of max)')
    for n, f in flops.items():
        print(f'{n}\t{f}\t({f/flops_max:.2%})')


if __name__ == '__main__':
    main()
