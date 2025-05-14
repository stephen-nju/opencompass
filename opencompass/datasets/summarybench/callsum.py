"""
通话摘要测试集
"""
import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class CallSumDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        conversations=[]
        with open(path, 'r', errors='ignore') as in_f:
            data = json.load(in_f)
            for doc in data:
                convs=doc["conversations"]
                title=doc["title"]
                abstract=doc["abstract"]
                keypoint=doc["key_point"]
                conv_data=[]
                for conv in convs:
                    speaker=conv["speaker"]
                    if speaker=="A":
                        speaker="我"
                    else:
                        speaker="对方"
                    value=conv["value"]
                    if isinstance(value,list):
                        value= " ,".join(value)
                    elif isinstance(value,str):
                        value=value
                    conv_data.append(f"{speaker}: {value}\n")
                conv_str=" ".join(conv_data)
                conversations.append({"conversations":conv_str,"title":title,"abstract":abstract,"key_point":keypoint})

        dataset = Dataset.from_dict(
            {'input': [d['conversations'] for d in conversations ],
            'output': [json.dumps({"标题":d['title'],"摘要":d['abstract'],"关键要点":d['key_point']},ensure_ascii=False) for d in conversations],
            })
        return dataset


@TEXT_POSTPROCESSORS.register_module('callsum')
def callsum_postprocess(text: str) -> str:
    # 使用jiebarouge 需要将原始文本的\n 替换
    text = text.replace("\n","")
    return text
