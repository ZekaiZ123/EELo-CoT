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

class WaitTokenScheduler:
    def __init__(self, alpha_new=0.05, alpha_repeat=0.005, threshold=0.8):
        self.alpha_new = alpha_new
        self.alpha_repeat = alpha_repeat
        self.threshold = threshold
        self.wait_prob = 0.0
        self.token_history = set()
        self.last_injected_idx = -1

    def update(self, token_id: int, tokenizer) -> float:
        token_str = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if not token_str:
            return self.wait_prob

        is_new = token_str not in self.token_history
        self.wait_prob += self.alpha_new if is_new else self.alpha_repeat
        self.token_history.add(token_str)
        return self.wait_prob

    def should_inject(self, input_ids, tokenizer) -> bool:
        return self.wait_prob > self.threshold

    def reset(self):
        self.wait_prob = 0.0

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
        self._sentence_cooldown = 4        # require 4 sentences in cooldown
        self.scheduler = WaitTokenScheduler(alpha_new=0.05, alpha_repeat=0.005, threshold=0.8)

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
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs, model_input_name="input_ids")


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


            # Update scheduler with current token
            token_id = next_tokens[0].item()
            _ = self.scheduler.update(token_id, self.tokenizer)

            # Check sentence boundary using _is_end_of_sentence
            if self._is_end_of_sentence(token_id):
                if self.scheduler.should_inject(input_ids, self.tokenizer):
                    cur_idx = len(input_ids[0])
                    if cur_idx > self.scheduler.last_injected_idx:
                        self.scheduler.last_injected_idx = cur_idx

                        # Inject "Wait"
                        extra_ids = self.tokenizer("Wait", add_special_tokens=False).input_ids
                        extra_tensor = torch.tensor(extra_ids, device=input_ids.device).unsqueeze(0)
                        input_ids = torch.cat([input_ids, extra_tensor], dim=1)

                        # Force forward pass to update caches
                        forced_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                        forced_out = model_forward(**forced_inputs, return_dict=True)

                        # Extract forced outputs for bookkeeping
                        forced_logits = forced_out.logits[:, -1, :].float()
                        forced_scores = logits_processor(input_ids, forced_logits)

                        if output_scores: scores += (forced_scores,)
                        if output_logits: raw_logits += (forced_logits,)
                        if output_attentions:
                            decoder_attentions += (forced_out.decoder_attentions
                                                if self.config.is_encoder_decoder
                                                else forced_out.attentions,)
                        if output_hidden_states:
                            decoder_hidden_states += (forced_out.decoder_hidden_states
                                                    if self.config.is_encoder_decoder
                                                    else forced_out.hidden_states,)

                        model_kwargs = self._update_model_kwargs_for_generation(
                            forced_out, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                        )

                        # Reset after successful injection
                        self.scheduler.reset()


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
