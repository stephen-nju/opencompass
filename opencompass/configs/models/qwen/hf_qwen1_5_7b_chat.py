from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen1.5-7b-chat-hf',
        path='/home/jovyan/zhubin/DATA/models/Qwen1.5-7B-Chat/',
        model_kwargs= dict(trust_remote_code=True),
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
