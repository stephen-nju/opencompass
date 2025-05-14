
from mmengine.config import read_base


summarizer = dict(
    dataset_abbrs=[
        ['lcsts','rouge1'],
        ['lcsts','rouge2'],
        ['lcsts','rougeL'],
        ['Xsum','rouge1'],
        ['Xsum','rouge2'],
        ['Xsum','rougeL'],
        ['cnewsum','rouge1'],
        ['cnewsum','rouge2'],
        ['cnewsum','rougeL'],
        ['csds','rouge1'],
        ['csds','rouge2'],
        ['csds','rougeL'],
        ['csl','rouge1'],
        ['csl','rouge2'],
        ['csl','rougeL'],
        ['vcsum','rouge1'],
        ['vcsum','rouge2'],
        ['vcsum','rougeL'],
        ['callsum','rouge1'],
        ['callsum','rouge2'],
        ['callsum','rougeL']
    ],
)
