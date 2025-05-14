
from mmengine.config import read_base

with read_base():
    # from ...ceval.ceval_gen_5f30c7 import ceval_datasets
    # from ...mmlu.mmlu_gen_4d595a import mmlu_datasets
    # from ...cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    #
    # from ...math.math_gen_265cce import math_datasets
    # from ...gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    # from ...bbh.bbh_gen_5b92b0 import bbh_datasets
    # from ...Xsum.Xsum_gen import Xsum_datasets
    # from ...lcsts.lcsts_gen import lcsts_datasets
    # from ...FewCLUE_csl.FewCLUE_csl_gen import csl_datasets
    # from ...summarybench.cnewsum.cnewsum_gen import cnewsum_datasets
    # from ...summarybench.csl.csl_gen import csl_datasets
    # from ...summarybench.vcsum.vcsum_gen import vcsum_datasets
    # from ...summarybench.csds.csds_gen import csds_datasets
    # from ...Xsum.Xsum_gen import Xsum_datasets
    # from ...lcsts.lcsts_gen import lcsts_datasets
    from ...summarybench.callsum.callsum_gen import callsum_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
