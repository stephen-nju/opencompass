

from opencompass.models import HuggingFaceCausalLM
from opencompass.models import HuggingFacewithChatTemplate
_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<TOKENS_UNUSED_1>'),
        dict(role='BOT', begin='<TOKENS_UNUSED_2>', generate=True),
    ],
    begin="<s>",
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='minicpm-4b-1epoch',
        path='/home/jovyan/zhubin/saved_checkpoint/minicpm_callsum_v3_ep3_lr2e5_bs4/checkpoint-952/',
        tokenizer_path='/home/jovyan/zhubin/saved_checkpoint/minicpm_callsum_v3_ep3_lr2e5_bs4/checkpoint-952/',
        max_out_len=512,
        max_seq_len=2048,
        batch_size=6,
        tokenizer_kwargs=dict(
                    model_max_length =2048,
                    padding_side = 'right',
                    use_fast = False,
                    trust_remote_code = True,
                    add_eos_token=False,
                    add_bos_token = False
        ),
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(trust_remote_code=True,device_map='auto'),
        meta_template=_meta_template,
    ),
    
    dict(
        type=HuggingFaceCausalLM,
        abbr='minicpm-4b-2epoch',
        path='/home/jovyan/zhubin/saved_checkpoint/minicpm_callsum_v3_ep3_lr2e5_bs4/checkpoint-1904/',
        tokenizer_path='/home/jovyan/zhubin/saved_checkpoint/minicpm_callsum_v3_ep3_lr2e5_bs4/checkpoint-1904/',
        max_out_len=512,
        max_seq_len=2048,
        batch_size=6,
        tokenizer_kwargs=dict(
                    model_max_length =2048,
                    padding_side = 'right',
                    use_fast = False,
                    trust_remote_code = True,
                    add_eos_token=False,
                    add_bos_token = False
        ),
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(trust_remote_code=True,device_map='auto'),
        meta_template=_meta_template,
    ),

    dict(
        type=HuggingFaceCausalLM,
        abbr='minicpm-4b-1epoch',
        path='/home/jovyan/zhubin/saved_checkpoint/minicpm_callsum_v3_ep3_lr2e5_bs4/checkpoint-2856/',
        tokenizer_path='/home/jovyan/zhubin/saved_checkpoint/minicpm_callsum_v3_ep3_lr2e5_bs4/checkpoint-2856',
        max_out_len=512,
        max_seq_len=2048,
        batch_size=6,
        tokenizer_kwargs=dict(
                    model_max_length =2048,
                    padding_side = 'right',
                    use_fast = False,
                    trust_remote_code = True,
                    add_eos_token=False,
                    add_bos_token = False
        ),
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(trust_remote_code=True,device_map='auto'),
        meta_template=_meta_template,
    )
]
