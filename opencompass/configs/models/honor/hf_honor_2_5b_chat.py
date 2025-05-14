
# from opencompass.models import HonorLM
from opencompass.models import HuggingFaceCausalLM
from opencompass.models import HuggingFacewithChatTemplate
_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<TOKENS_UNUSED_1>'),
        dict(role='BOT', begin='<TOKENS_UNUSED_2>', generate=True),
    ],
    begin="<s>",
)

#model_path="/home/jovyan/zhubin/saved_checkpoint/less_top0.05_magiclm_nano_conv_sum_v2_full_lr2e5_3epoch_bs4/"
#model_path="/home/jovyan/zhubin/saved_checkpoint/less_fb30_magiclm_nano_conv_sum_v2_full_lr2e5_3epoch_bs4/"
#model_path="/home/jovyan/zhubin/saved_checkpoint/less_top0.05_magiclm_nano_conv_sum_v2_lora_lr2e5_3epoch_bs4/merge"
#model_path="/home/jovyan/zhubin/DATA/models/MagicLM-3B-Instruct-v0.1/"
# model_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_lesstop0.05_lr2e5_2epoch_bs4/"
# model_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_v0.1_vanllia_full_lr5e6_1epoch_bs4"
#model_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_lesstop0.05_lr2e5_1epoch_bs4"
#model_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_lesstop0.2_lr2e5_1epoch_bs4"

#model_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_lesstop0.2_lr2e5_2epoch_bs4"
#model_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_v0.1_callsum_lr2e5_2epoch_bs4/"
model_path="/home/jovyan/zhubin/saved_checkpoint/0925_magiclm_nano_neft_clv4_ups_dialog_ep3_lr2e5_bs4/checkpoint-3145/"
model_path2="/home/jovyan/zhubin/saved_checkpoint/0925_magiclm_nano_neft_clv4_ups_dialog_ep3_lr2e5_bs4/checkpoint-6290/"
model_path3="/home/jovyan/zhubin/saved_checkpoint/0925_magiclm_nano_neft_clv4_ups_dialog_ep3_lr2e5_bs4/checkpoint-9435/"
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='magiclm nano 1epoch',
        path=model_path,
        tokenizer_path=model_path,
        max_out_len=512,
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
        meta_template=_meta_template,
    ),
    dict(
        type=HuggingFaceCausalLM,
        abbr='magiclm nano 2epoch',
        path=model_path2,
        tokenizer_path=model_path2,
        max_out_len=512,
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
        meta_template=_meta_template,
    ),

    dict(
        type=HuggingFaceCausalLM,
        abbr='magiclm nano 3epoch',
        path=model_path3,
        tokenizer_path=model_path3,
        max_out_len=512,
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
        meta_template=_meta_template,
    )
]
    # dict(
    #     type=HuggingFaceCausalLM,
    #     abbr='magiclm nano 1epoch',
    #     path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_callsum_v3_ep3_lr2e5_bs4/checkpoint-952/",
    #     tokenizer_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_callsum_v3_ep3_lr2e5_bs4/checkpoint-952/",
    #     max_out_len=512,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     tokenizer_kwargs=dict(
    #                 model_max_length =2048,
    #                 padding_side = 'right',
    #                 use_fast = False,
    #                 trust_remote_code = True,
    #                 add_eos_token=False,
    #                 add_bos_token = False
    # ),
    #     model_kwargs=dict(trust_remote_code=True,device_map='auto'),
    #     run_cfg=dict(num_gpus=1),
    #     meta_template=_meta_template,
    # ),
    # dict(
    #     type=HuggingFaceCausalLM,
    #     abbr='magiclm nano 2epoch',
    #     path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_callsum_v3_ep3_lr2e5_bs4/checkpoint-1904/",
    #     tokenizer_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_callsum_v3_ep3_lr2e5_bs4/checkpoint-1904/",
    #     max_out_len=512,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     tokenizer_kwargs=dict(
    #                 model_max_length =2048,
    #                 padding_side = 'right',
    #                 use_fast = False,
    #                 trust_remote_code = True,
    #                 add_eos_token=False,
    #                 add_bos_token = False
    # ),
    #     model_kwargs=dict(trust_remote_code=True,device_map='auto'),
    #     run_cfg=dict(num_gpus=1),
    #     meta_template=_meta_template,
    # ),

    # dict(
    #     type=HuggingFaceCausalLM,
    #     abbr='magiclm nano 3epoch',
    #     path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_callsum_v3_ep3_lr2e5_bs4/checkpoint-2856/",
    #     tokenizer_path="/home/jovyan/zhubin/saved_checkpoint/magiclm_nano_callsum_v3_ep3_lr2e5_bs4/checkpoint-2856/",
    #     max_out_len=512,
    #     max_seq_len=2048,
    #     batch_size=8,
    #     tokenizer_kwargs=dict(
    #                 model_max_length =2048,
    #                 padding_side = 'right',
    #                 use_fast = False,
    #                 trust_remote_code = True,
    #                 add_eos_token=False,
    #                 add_bos_token = False
    # ),
    #     model_kwargs=dict(trust_remote_code=True,device_map='auto'),
    #     run_cfg=dict(num_gpus=1),
    #     meta_template=_meta_template,
    # ),

