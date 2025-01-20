
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.summarybench import CSDSDataset
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator


csds_reader_cfg = dict(input_columns=['input'], output_column='output')

csds_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt='下面是一段客服和用户的对话，请对对话内容做个摘要: {input}\n'),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

csds_eval_cfg = dict(
    evaluator=dict(type=JiebaRougeEvaluator),
    pred_role='BOT',
    # pred_postprocessor=dict(type='csds'),
)

csds_datasets = [
    dict(
        type=CSDSDataset,
        abbr='csds',
        path='./data/summarybench/CSDS/csds_dialogue_format_dev.json',
        reader_cfg=csds_reader_cfg,
        infer_cfg=csds_infer_cfg,
        eval_cfg=csds_eval_cfg,
    )
]
