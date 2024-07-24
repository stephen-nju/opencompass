import sys
from typing import Dict, List, Optional

import numpy as np
import torch

from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.registry import MODELS


@MODELS.register_module()
class HonorLM(HuggingFaceCausalLM):
    # honor 2.5b 模型测评
    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):

        sys.path.append(tokenizer_path)
        from tokenization_moss2 import Moss2Tokenizer
        self.tokenizer = Moss2Tokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs)

    def _load_model(self,
                    path: str,
                    model_kwargs: dict,
                    peft_path: Optional[str] = None):

        sys.path.append(path)
        from modeling_moss2 import Moss2ForCausalLM
        self.model = Moss2ForCausalLM.from_pretrained(path, **model_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    # def generate(self,
    #              inputs: List[str],
    #              max_out_len: int,
    #              min_out_len: Optional[int] = None,
    #              stopping_criteria: List[str] = [],
    #              **kwargs) -> List[str]:
    #     """Generate results given a list of inputs.

    #     Args:
    #         inputs (List[str]): A list of strings.
    #         max_out_len (int): The maximum length of the output.
    #         min_out_len (Optional[int]): The minimum length of the output.

    #     Returns:
    #         List[str]: A list of generated strings.
    #     """
    #     generation_kwargs = kwargs.copy()
    #     generation_kwargs.update(self.generation_kwargs)
    #     if self.batch_padding and len(inputs) > 1:
    #         return self._batch_generate(inputs=inputs,
    #                                     max_out_len=max_out_len,
    #                                     min_out_len=min_out_len,
    #                                     stopping_criteria=stopping_criteria,
    #                                     **generation_kwargs)
    #     else:
    #         return sum(
    #             (self._single_generate(inputs=[input_],
    #                                    max_out_len=max_out_len,
    #                                    min_out_len=min_out_len,
    #                                    stopping_criteria=stopping_criteria,
    #                                    **generation_kwargs)
    #              for input_ in inputs), [])

    # def _batch_generate(self,
    #                     inputs: List[str],
    #                     max_out_len: int,
    #                     min_out_len: Optional[int] = None,
    #                     stopping_criteria: List[str] = [],
    #                     **kwargs) -> List[str]:
    #     """Support for batch prompts inference.

    #     Args:
    #         inputs (List[str]): A list of strings.
    #         max_out_len (int): The maximum length of the output.

    #     Returns:
    #         List[str]: A list of generated strings.
    #     """
    #     if self.extract_pred_after_decode:
    #         prompt_lens = [len(input_) for input_ in inputs]

    #     if self.use_fastchat_template:
    #         try:
    #             from fastchat.model import get_conversation_template
    #         except ModuleNotFoundError:
    #             raise ModuleNotFoundError(
    #                 'Fastchat is not implemented. You can use '
    #                 '\'pip install "fschat[model_worker,webui]"\' '
    #                 'to implement fastchat.')
    #         for i in range(len(inputs)):
    #             conv = get_conversation_template('vicuna')
    #             conv.append_message(conv.roles[0], inputs[i])
    #             conv.append_message(conv.roles[1], None)
    #             inputs[i] = conv.get_prompt()

    #     # step-1: tokenize the input with batch_encode_plus
    #     tokens = self.tokenizer.batch_encode_plus(inputs,
    #                                               padding=True,
    #                                               truncation=True,
    #                                               max_length=self.max_seq_len)
    #     tokens = {
    #         k: torch.tensor(np.array(tokens[k]), device=self.model.device)
    #         for k in tokens if k in ['input_ids', 'attention_mask']
    #     }

    #     origin_stopping_criteria = stopping_criteria
    #     if stopping_criteria:
    #         # Construct huggingface stopping criteria
    #         if self.tokenizer.eos_token is not None:
    #             stopping_criteria = stopping_criteria + [
    #                 self.tokenizer.eos_token
    #             ]
    #         stopping_criteria = transformers.StoppingCriteriaList([
    #             *[
    #                 MultiTokenEOSCriteria(sequence, self.tokenizer,
    #                                       tokens['input_ids'].shape[0])
    #                 for sequence in stopping_criteria
    #             ],
    #         ])
    #         kwargs['stopping_criteria'] = stopping_criteria

    #     if min_out_len is not None:
    #         kwargs['min_new_tokens'] = min_out_len

    #     # step-2: conduct model forward to generate output
    #     outputs = self.model.generate(**tokens,
    #                                   max_new_tokens=max_out_len,
    #                                   **kwargs)

    #     if not self.extract_pred_after_decode:
    #         outputs = outputs[:, tokens['input_ids'].shape[1]:]

    #     decodeds = self.tokenizer.batch_decode(outputs,
    #                                            skip_special_tokens=True)

    #     if self.extract_pred_after_decode:
    #         decodeds = [
    #             token[len_:] for token, len_ in zip(decodeds, prompt_lens)
    #         ]

    #     if self.end_str:
    #         decodeds = [token.split(self.end_str)[0] for token in decodeds]
    #     if origin_stopping_criteria:
    #         for t in origin_stopping_criteria:
    #             decodeds = [token.split(t)[0] for token in decodeds]
    #     return decodeds

    # def _single_generate(self,
    #                      inputs: List[str],
    #                      max_out_len: int,
    #                      min_out_len: Optional[int] = None,
    #                      stopping_criteria: List[str] = [],
    #                      **kwargs) -> List[str]:
    #     """Support for single prompt inference.

    #     Args:
    #         inputs (List[str]): A list of strings.
    #         max_out_len (int): The maximum length of the output.

    #     Returns:
    #         List[str]: A list of generated strings.
    #     """
    #     if self.extract_pred_after_decode:
    #         prompt_lens = [len(input_) for input_ in inputs]

    #     if self.use_fastchat_template:
    #         try:
    #             from fastchat.model import get_conversation_template
    #         except ModuleNotFoundError:
    #             raise ModuleNotFoundError(
    #                 'Fastchat is not implemented. You can use '
    #                 '\'pip install "fschat[model_worker,webui]"\' '
    #                 'to implement fastchat.')
    #         conv = get_conversation_template('vicuna')
    #         conv.append_message(conv.roles[0], inputs[0])
    #         conv.append_message(conv.roles[1], None)
    #         inputs = [conv.get_prompt()]

    #     if self.mode == 'mid':
    #         input_ids = self.tokenizer(inputs, truncation=False)['input_ids']
    #         input_ids = torch.tensor(input_ids, device=self.model.device)
    #         if len(input_ids[0]) > self.max_seq_len - max_out_len:
    #             half = int((self.max_seq_len - max_out_len) / 2)
    #             inputs = [
    #                 self.tokenizer.decode(input_ids[0][:half],
    #                                       skip_special_tokens=True) +
    #                 self.tokenizer.decode(input_ids[0][-half:],
    #                                       skip_special_tokens=True)
    #             ]

    #     input_ids = self.tokenizer(inputs,
    #                                truncation=True,
    #                                max_length=self.max_seq_len -
    #                                max_out_len)['input_ids']
    #     input_ids = torch.tensor(input_ids, device=self.model.device)
    #     origin_stopping_criteria = stopping_criteria
    #     if stopping_criteria:
    #         # Construct huggingface stopping criteria
    #         if self.tokenizer.eos_token is not None:
    #             stopping_criteria = stopping_criteria + [
    #                 self.tokenizer.eos_token
    #             ]
    #         stopping_criteria = transformers.StoppingCriteriaList([
    #             *[
    #                 MultiTokenEOSCriteria(sequence, self.tokenizer,
    #                                       input_ids.shape[0])
    #                 for sequence in stopping_criteria
    #             ],
    #         ])
    #         kwargs['stopping_criteria'] = stopping_criteria

    #     if min_out_len is not None:
    #         kwargs['min_new_tokens'] = min_out_len

    #     # To accommodate the PeftModel, parameters should be passed in
    #     # key-value format for generate.
    #     outputs = self.model.generate(input_ids=input_ids,
    #                                   max_new_tokens=max_out_len,
    #                                   **kwargs)

    #     if not self.extract_pred_after_decode:
    #         outputs = outputs[:, input_ids.shape[1]:]

    #     decodeds = self.tokenizer.batch_decode(outputs,
    #                                            skip_special_tokens=True)

    #     if self.extract_pred_after_decode:
    #         decodeds = [
    #             token[len_:] for token, len_ in zip(decodeds, prompt_lens)
    #         ]

    #     if self.end_str:
    #         decodeds = [token.split(self.end_str)[0] for token in decodeds]
    #     if origin_stopping_criteria:
    #         for t in origin_stopping_criteria:
    #             decodeds = [token.split(t)[0] for token in decodeds]
    #     return decodeds
