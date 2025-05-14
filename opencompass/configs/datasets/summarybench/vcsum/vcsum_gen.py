
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.summarybench import VCSumDataset
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator,ChineseRougeEvaluator


vcsum_reader_cfg = dict(input_columns=['instruction','input'], output_column='output')

vcsum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{instruction} {input}'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

vcsum_eval_cfg = dict(
    evaluator=dict(type=ChineseRougeEvaluator),
    pred_role='BOT',
    # pred_postprocessor=dict(type='vcsum'),
)

vcsum_datasets = [
    dict(
        type=VCSumDataset,
        abbr='vcsum',
        path='./data/summarybench/VCSum/vcsum_headlines_dev_part0.json',
        reader_cfg=vcsum_reader_cfg,
        infer_cfg=vcsum_infer_cfg,
        eval_cfg=vcsum_eval_cfg,
    )
]
