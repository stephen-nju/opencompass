
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.summarybench import CallSumDataset
from opencompass.openicl.icl_evaluator import JiebaRougeEvaluator,ChineseRougeEvaluator
from opencompass.utils.text_postprocessors import general_cn_postprocess

callsum_reader_cfg = dict(input_columns=['input'], output_column='output')

gen_prompt="""你是一个手机通话摘要助手，你的任务是根据用户的通话记录生成通话摘要。你需要遵循以下要求完成任务。

##人设理解
1. 我会在通话记录中给出通话双方的称呼，你需要根据通话记录中的内容，推断通话双方的人物关系和称呼。
2. 请使用合适的称呼来称呼对方。如果无法推断，则使用“我”和“对方”来表示人物关系。

## 内容要求
1. 简洁性：摘要内容整体表达简洁易懂，有概括性
2. 事实性：摘要的内容要基于通话记录中的内容，不能出现不符合事实的描述。请确保你的摘要总结是基于通话记录的内容，避免添加任何外部信息或假设。
3. 连贯性：摘要的内容和逻辑需要保持连贯，在文字表达上需要清晰，易懂，流畅，自然，方便用户轻松理解
4. 重点内容：摘要的内容需要保证通话记录中重要信息不被遗漏。

##格式要求
1. 一份完整的通话摘要通常包含三个部分，标题，摘要和关键要点。
2. 标题形式上是一个短语描述，标题中不能出现标点符号。
3. 关键要点需要分别描述

##风格要求
1. 口语化表达：采用偏口语化表达，贴近日常生活，避免使用非常正式的表达方式。
2. 干练表达：避免出现啰嗦的情况。

##注意事项
1. 请使用第一人称视角，完成摘要生成
2. 摘要的目标是生成一个准确、简洁且全面的通话摘要。确保用户能够迅速且全面地了解通话的主要内容和细节。


##输出格式
- 完整的通话摘要需要包含两个部分：标题和摘要
- 请以JSON格式输出结果，格式要求如下：{\"标题\": \"这里填写标题内容\",\"摘要\": \"这里填写摘要内容\",\"关键要点\":\"这里填写关键要点内容\"}

##通话记录：
{input}

你需要仔细阅读通话记录，根据上述要求生成结果：
"""

callsum_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=gen_prompt),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

callsum_eval_cfg = dict(
    evaluator=dict(type=ChineseRougeEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type='callsum'),# 预测结果的后处理
    # dataset_postprocessor=dict(type=general_cn_postprocess)) # 数据集标准答案的后处理
)

callsum_datasets = [
    dict(
        type=CallSumDataset,
        abbr='callsum',
        path='./data/summarybench/CallSum/union_conversations_dev_v2.json',
        reader_cfg=callsum_reader_cfg,
        infer_cfg=callsum_infer_cfg,
        eval_cfg=callsum_eval_cfg,
    )
]
