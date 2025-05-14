
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.summarybench import CSLDataset
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator,ChineseRougeEvaluator


csl_reader_cfg = dict(input_columns=['instruction','input'], output_column='output')

csl_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='{instruction} {input}\n'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

csl_eval_cfg = dict(
    evaluator=dict(type=ChineseRougeEvaluator),
    pred_role='BOT',
    # pred_postprocessor=dict(type='csl'),
)

csl_datasets = [
    dict(
        type=CSLDataset,
        abbr='csl',
        path='./data/summarybench/CSL/csl_kg_ts_benchmark_dev.json',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg,
    )
]
