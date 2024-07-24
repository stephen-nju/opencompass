from opencompass.models import HonorLM
from opencompass.models import HuggingFaceBaseModel
# _meta_template = dict(

#     round=[
#         dict(role='HUMAN', begin='<TOKENS_UNUSED_1>'),
#         dict(role='BOT', begin='<TOKENS_UNUSED_2>', generate=True),
#     ],
# )
# models = [
#     dict(
#         type=HuggingFaceCausalLM,
#         abbr='honor_2_5b_chat',
#         path='/home/jovyan/zhubin/saved_checkpoint/honor_2.5b_conv_abstract_v2_lora64_lr6e5_3epoch_bs4/merge/',
#         tokenizer_path='/home/jovyan/zhubin/saved_checkpoint/honor_2.5b_conv_abstract_v2_lora64_lr6e5_3epoch_bs4/merge',
#         max_out_len=512,
#         max_seq_len=2048,
#         batch_size=8,
#         tokenizer_kwargs=dict(
#                     model_max_length =2048,
#                     padding_side = "right",
#                     use_fast = False,
#                     trust_remote_code = True,
#                     add_eos_token=False,
#                     add_bos_token = False
#     ),
#         model_kwargs=dict(trust_remote_code=True,device_map="auto"),
#         run_cfg=dict(num_gpus=1),
#         meta_template=_meta_template,
#     )
# ]


models = [
    dict(
        type=HonorLM,
        abbr='honor_2_5b_base',
        path='/home/jovyan/zhubin/DATA/models/honor2_5b_patch',
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=8,
        tokenizer_kwargs=dict(
                    model_max_length =2048,
                    padding_side = 'right',
                    use_fast = False,
                    trust_remote_code = True,
                    add_eos_token=False,
                    add_bos_token = False
    ),

        model_kwargs=dict(trust_remote_code=True,device_map='auto'),
        run_cfg=dict(num_gpus=1),
    )
]
