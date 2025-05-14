
from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-3b-1epoch',
        path='/home/jovyan/zhubin/saved_checkpoint/0925_Qwen2.5_3b_instruct_neft_alp_dig_callsumv3_ep3_lr2e5_bs4/checkpoint-1513/',
        max_out_len=512,
        max_seq_len=2048,
        batch_size=4,
        model_kwargs=dict(trust_remote_code=True,torch_dtype="auto",device_map="auto"),
        tokenizer_kwargs=dict(
            use_fast=False,
            trust_remote_code=True
        ),
        run_cfg=dict(num_gpus=1),
    ),

    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-3b-2epoch',
        path='/home/jovyan/zhubin/saved_checkpoint/0925_Qwen2.5_3b_instruct_neft_alp_dig_callsumv3_ep3_lr2e5_bs4/checkpoint-3027/',
        max_out_len=512,
        max_seq_len=2048,
        batch_size=4,
        model_kwargs=dict(trust_remote_code=True,torch_dtype="auto",device_map="auto"),
        tokenizer_kwargs=dict(
            use_fast=False,
            trust_remote_code=True
        ),
        run_cfg=dict(num_gpus=1),
    ),

    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-3b-3epoch',
        path='/home/jovyan/zhubin/saved_checkpoint/0925_Qwen2.5_3b_instruct_neft_alp_dig_callsumv3_ep3_lr2e5_bs4/checkpoint-4539/',
        max_out_len=512,
        max_seq_len=2048,
        batch_size=4,
        model_kwargs=dict(trust_remote_code=True,torch_dtype="auto",device_map="auto"),
        tokenizer_kwargs=dict(
            use_fast=False,
            trust_remote_code=True
        ),
        run_cfg=dict(num_gpus=1),
    )
]
