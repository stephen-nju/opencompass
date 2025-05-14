
from mmengine.config import read_base

with read_base():

    # from .models.honor.hf_hornor_2_5B_base import models
    # from .models.qwen.hf_qwen2_1_5b_instruct import models as qwen2_1_5b_models
    # from .models.qwen.hf_qwen2_7b_instruct import models
    # from .models.qwen.vllm_qwen2_1_5b_instruct import models
    from .models.honor.hf_honor_2_5b_chat import models as hf_honor_2_5b_chat_models
    # from .models.openbmb.hf_minicpm_4b_honor import models as hf_minicpn_4b_chat_models
    # from .models.qwen.hf_qwen_7b_chat import models
    # from .models.qwen.lmdeploy_qwen2_7b_instruct import models
    # from .models.qwen.vllm_qwen2_72b_instruct import models
    # from .models.qwen.hf_qwen1_5_7b_chat import models
    # from .models.qwen.hf_qwen2_5_3b_instruct import models as qwen2_5_3b_models

    from .datasets.collections.leaderboard.honor_chat import datasets
    from .summarizers.summarybench import summarizer

models= sum((v for k, v in locals().items() if k.endswith('_models')), [])


