import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class CSLDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, 'r', errors='ignore') as in_f:
            data = json.load(in_f)
            dataset = Dataset.from_dict({
                'instruction': [d['instruction'] for d in data],
                'input': [d['input'] for d in data],
                'output': [d['output'] for d in data]
            })
            return dataset


# @TEXT_POSTPROCESSORS.register_module('csl')
# def csl_postprocess(text: str) -> str:
#     text = text.strip().split('\n')[0].strip()
#     return text
