import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class VCSumDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        with open(path, 'r', errors='ignore') as in_f:
            data = json.load(in_f)
            # rows = []
            # for i, line in enumerate(in_f):
            #     if i == 1000:
            #         break
            #     sample = json.loads(line.strip())
            #     dialogue = sample['dialogue']
            #     summary = sample['summary']
            #     if isinstance(dialogue, float) or isinstance(summary, float):
            #         continue
            #     rows.append({'dialogue': dialogue, 'summary': summary})
            dataset = Dataset.from_dict({
                'instruction': [d['instruction'] for d in data],
                'input': [d['input'] for d in data],
                'output': [d['output'] for d in data]
            })
            return dataset


@TEXT_POSTPROCESSORS.register_module('vcsum')
def vcsum_postprocess(text: str) -> str:
    text = text.strip().split('\n')[0].strip()
    return text
