
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.summarybench import CNewSumDataset
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator,ChineseRougeEvaluator

cnewsum_reader_cfg = dict(input_columns=['input'], output_column='output')

cnewsum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='生成下面这段新闻摘要。{input}\n'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

cnewsum_eval_cfg = dict(
    evaluator=dict(type=ChineseRougeEvaluator),
    pred_role='BOT',
    # pred_postprocessor=dict(type='cnewsum'),
)

cnewsum_datasets = [
    dict(
        type=CNewSumDataset,
        abbr='cnewsum',
        path='./data/summarybench/CNewSum/cnewsum_dev_2000.json',
        reader_cfg=cnewsum_reader_cfg,
        infer_cfg=cnewsum_infer_cfg,
        eval_cfg=cnewsum_eval_cfg,
    )
]
