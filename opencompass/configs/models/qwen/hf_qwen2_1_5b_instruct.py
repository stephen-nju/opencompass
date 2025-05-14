from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2-1.5b-instruct-hf',
        path='/home/jovyan/zhubin/DATA/models/Qwen2-1.5B-Instruct/',
        max_out_len=1024,
        batch_size=4,
        model_kwargs=dict(trust_remote_code=True,torch_dtype="auto",device_map="auto"),
        tokenizer_kwargs=dict(
            use_fast=False,
            trust_remote_code=True
        ),
        run_cfg=dict(num_gpus=1),
    )
]
