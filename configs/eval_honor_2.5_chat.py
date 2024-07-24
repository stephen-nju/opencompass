
from mmengine.config import read_base

with read_base():

    # from .models.honor.hf_hornor_2_5B_base import models
    # from .models.qwen.hf_qwen2_1_5b_instruct import models
    # from .models.qwen.hf_qwen2_7b_instruct import models
    # from .models.qwen.vllm_qwen2_1_5b_instruct import models
    # from .models.honor.hf_honor_2_5b_chat import models
    # from .models.qwen.hf_qwen_7b_chat import models
    from .models.qwen.lmdeploy_qwen2_7b_instruct import models


    from .datasets.collections.leaderboard.honor_chat import datasets
    from .summarizers.summarybench import summarizer
