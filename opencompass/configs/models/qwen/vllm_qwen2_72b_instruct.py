from opencompass.models import VLLMwithChatTemplate

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen2-72b-instruct-vllm',
        path='/home/jovyan/zhubin/DATA/models/Qwen2-72B-instruct/',
        model_kwargs=dict(tensor_parallel_size=4),
        max_out_len=1024,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
    )
]
