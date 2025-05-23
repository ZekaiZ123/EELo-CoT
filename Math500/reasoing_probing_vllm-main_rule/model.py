import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sympy as sp
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import math
from transformers import Qwen2ForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Union, List

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    HybridChunkedCache,
    OffloadedCache,
    QuantizedCacheConfig,
    StaticCache,
)
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)
from transformers.generation.streamers import BaseStreamer
import os

class DeepSeekQwenModel(Qwen2ForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        self.recent_text = ""
        self.digit_count = 0
        self.last_period_idx = 0
        self.min_digits        = 8
        self.last_injected_idx  = -1    # marker so we only inject once per sentence
        self.armed = True                  # ready to fire
        self.sentences_since_trigger = 0   # how many sentences have passed since last fire
        self._sentence_cooldown = 5        # require 5 sentences in cooldown

        # Hook & schedule state
        if "intervene_functions" in kwargs:
            self.intervene_functions = kwargs.pop("intervene_functions")  # pop to prevent serialization issues
            
            # Set the module for each intervention function
            layer_idx = kwargs.pop("layer_idx", 27)  # Default to layer 27 if not specified, pop to prevent serialization
            for intervene_function in self.intervene_functions:
                # intervene_function.module = self.model.layers[layer_idx].mlp.act_fn
                modules = [self.model.layers[i].mlp.act_fn for i in range(28)]
                intervene_function.modules = modules

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override from_pretrained to handle intervention functions properly"""
        
        # Store intervene_functions and layer_idx before calling parent method
        intervene_functions = kwargs.pop("intervene_functions", None)
        layer_idx = kwargs.pop("layer_idx", 27)
        
        # Call parent method
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Restore intervene_functions after parent call
        if intervene_functions is not None:
            model.intervene_functions = intervene_functions
            for intervene_function in model.intervene_functions:
                # intervene_function.module = model.model.layers[layer_idx].mlp.act_fn
                modules = [model.model.layers[i].mlp.act_fn for i in range(28)]
                intervene_function.modules = modules
        
        return model

    def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
        return False
    
    def _is_end_of_sentence(self, token_id: int) -> bool:
        token_text = self.tokenizer.decode([token_id])
        context    = self.recent_text + token_text
        has_ending_punct = re.search(r"[.!?。！？]$", token_text.strip()) is not None

        if has_ending_punct:
            # rule out decimals like “3.14”
            if re.search(r"\d\.\d*$", token_text): 
                return False
            # rule out “2.” in formulas
            if re.search(r"\d\.$", token_text) and re.search(r"[-+*/=><…]\s*\d+\.$", context[-20:]):
                return False
            # math functions, abbreviations, ellipses, LaTeX…
            if re.search(r"(sin|cos|tan|log|…)\s*\([^)]*\d+\.\d*", context[-30:], re.I):
                return False
            if re.search(r"(Mr\.|Mrs\.|Dr\.|etc\.|e\.g\.)$", token_text, re.I):
                return False
            if "..." in token_text or ".." in token_text:
                return False
            if context.count("$") % 2 == 1 or context.count(r"\begin{equation}") > context.count(r"\end{equation}"):
                return False
            return True

        # newline‐based endings
        if re.search(r"\n$", token_text) and not re.search(r"\\\n$", context[-5:]):
            if context.count("```") % 2 == 0:
                return True

        # “!?” always end
        if re.search(r"[!?。！？]$", token_text):
            return True

        return False
    
    from typing import List


    def check_trigger_from_last_period(
        self,
        token_ids: List[int],
        min_digits: int = 5
    ):
        cur_idx = len(token_ids) - 1
        tok = self.tokenizer.decode([token_ids[cur_idx]], skip_special_tokens=True)

        # detect end-of-sentence punctuation
        if any(p in tok for p in [".", "!", "?", ":"]):
            # slice out the just-completed sentence
            slice_ids = token_ids[self.last_period_idx : cur_idx + 1]
            text_slice = self.tokenizer.decode(slice_ids, skip_special_tokens=True)
            digit_count = sum(ch.isdigit() for ch in text_slice)

            if not self.armed:
                # still cooling down: count this sentence, do NOT fire
                self.sentences_since_trigger += 1
                if self.sentences_since_trigger >= self._sentence_cooldown:
                    # cooldown complete → re-arm
                    self.armed = True
                trigger = False

            else:
                # armed → check digit criterion
                if digit_count > min_digits:
                    trigger = True
                    # immediately disarm and reset cooldown counter
                    self.armed = False
                    self.sentences_since_trigger = 0
                else:
                    trigger = False
                    # remain armed for next sentence if you prefer

            # advance window start so next slice is correct
            new_start = cur_idx + 1
            self.last_period_idx = new_start
            return trigger, new_start

        # not a sentence boundary → nothing changes
        return False, self.last_period_idx

    
    # def check_trigger_from_last_period(
    #     self,
    #     token_ids: List[int],
    #     min_digits: int = 5
    # ):
    #     cur_idx = len(token_ids) - 1
    #     tok = self.tokenizer.decode([token_ids[cur_idx]], skip_special_tokens=True)

    #     # did we just hit end-of-sentence?
    #     if "." in tok or "!" in tok or "?" in tok or ":" in tok:
    #         # grab everything since last period (inclusive) up to here
    #         slice_ids = token_ids[self.last_period_idx : cur_idx + 1]
    #         text_slice = self.tokenizer.decode(slice_ids, skip_special_tokens=True)
    #         digit_count = sum(ch.isdigit() for ch in text_slice)

    #         if self.triggered:
    #             # we triggered on the previous sentence → skip this one, but now re-arm
    #             trigger = False
    #             self.triggered = False
    #         else:
    #             # normal check
    #             trigger = digit_count > min_digits
    #             if trigger:
    #                 self.triggered = True

    #         # advance the window
    #         new_start = cur_idx + 1
    #         self.last_period_idx = new_start
    #         return trigger, new_start

    #     # not end of sentence → nothing changes
    #     return False, self.last_period_idx

    # def check_trigger_from_last_period(
    #     self,
    #     token_ids: List[int],
    #     min_digits: int = 5
    # ):
    #     cur_idx = len(token_ids) - 1
    #     tok = self.tokenizer.decode([token_ids[cur_idx]], skip_special_tokens=True)
    #     if "." in tok or "!" in tok or "?" in tok or ":" in tok:
    #         # 解码从上次句号后到当前句号的全部文本
    #         slice_ids = token_ids[self.last_period_idx : cur_idx + 1]
    #         text_slice = self.tokenizer.decode(slice_ids, skip_special_tokens=True)


    #         digit_count = sum(ch.isdigit() for ch in text_slice)

    #         trigger = digit_count > min_digits

    #         new_start = cur_idx + 1
    #         return trigger, new_start
    #     else:
    #         return False, self.last_period_idx
        

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):



            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # Apply intervention functions on the generated token
            for intervene_function in self.intervene_functions:
                intervene_function(next_tokens[0].item())
                
            
            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)


# ==========================================================
            # # 3. 检查是否需要插入 “Wait…” 
            # last_token_id = input_ids[0, -1].item()
            # # check sentence end
            # if self._is_end_of_sentence(last_token_id):
            #     # decode and update context
            #     text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            #     self.recent_text = text

            #     # grab last sentence and count digits
            #     prev_sent = text.rstrip().rsplit('.', 1)[-1]
            #     self.digit_count = sum(ch.isdigit() for ch in prev_sent)
            
            # if self.digit_count > 5:

            #all_ids = input_ids[0].tolist()

            # 判断是否触发插入
            # do_insert, self.last_period_idx = self.check_trigger_from_last_period(
            #     token_ids=all_ids,
            #     min_digits=5
            # )
            # only consider injection if the model just emitted TWO newlines
            # newline_ids = self.tokenizer("\n\n", add_special_tokens=False).input_ids
            # if len(all_ids) >= len(newline_ids) and all_ids[-len(newline_ids):] == newline_ids:
            #     do_insert, self.last_period_idx = self.check_trigger_from_last_period(
            #         token_ids=all_ids,
            #         min_digits=self.min_digits
            #     )
            
            # else:
            #     do_insert = False
            full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            if full_text.endswith("\n\n"):
                # we've just started a fresh paragraph → run digit trigger
                all_ids = input_ids[0].tolist()
                do_insert, self.last_period_idx = self.check_trigger_from_last_period(
                    token_ids=all_ids,
                    min_digits=self.min_digits
                )
            else:
                do_insert = False
                
            if do_insert:
                # 准备强制的文本及其 ID
                #extra = "Wait"
                # Make sure we haven’t already injected at this same spot
                cur_idx = len(all_ids)
                if cur_idx <= self.last_injected_idx:
                    do_insert = False
                else:
                    self.last_injected_idx = cur_idx
                # now inject “Wait” 
                extra     = "Wait"
                
                extra_ids = self.tokenizer(extra, add_special_tokens=False).input_ids
                extra_tensor = torch.tensor(extra_ids, device=input_ids.device).unsqueeze(0)

                # 3a. 先更新 input_ids
                input_ids = torch.cat([input_ids, extra_tensor], dim=1)

                # 3b. 重新 prepare inputs，并做一次“真实”前向
                forced_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                forced_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
                forced_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
                forced_out = model_forward(**forced_inputs, return_dict=True)

                # 3c. 提取 forced_out 对应 step 的中间量
                forced_logits = forced_out.logits[:, -1, :].float()
                forced_scores = logits_processor(input_ids, forced_logits)
                forced_attn   = (forced_out.decoder_attentions 
                                if self.config.is_encoder_decoder else forced_out.attentions)
                forced_hs     = (forced_out.decoder_hidden_states 
                                if self.config.is_encoder_decoder else forced_out.hidden_states)
                forced_cross  = getattr(forced_out, "cross_attentions", None)

                # 3d. 把这一 token 的数据 append 到记录里
                if output_scores:     scores            += (forced_scores,)
                if output_logits:     raw_logits        += (forced_logits,)
                if output_attentions: 
                    decoder_attentions += (forced_attn,)
                    if forced_cross is not None:
                        cross_attentions  += (forced_cross,)
                if output_hidden_states:
                    decoder_hidden_states += (forced_hs,)

                # 3e. 更新缓存，保证后续生成连贯
                model_kwargs = self._update_model_kwargs_for_generation(
                    forced_out, model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder
                )
# ==========================================================

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids
