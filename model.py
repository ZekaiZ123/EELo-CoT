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
import random
import math

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
        self._sentence_cooldown = 4        # require 4 sentences in cooldown
        self.wait_cyclical_amplitude = kwargs.pop("wait_cyclical_amplitude", 3.0)
        self.wait_cyclical_period = kwargs.pop("wait_cyclical_period", 100.0)
        self.wait_cyclical_shift = kwargs.pop("wait_cyclical_shift", 0.0)
        self.reflection_cooldown = 4
        self.sentences_since_reflection = self.reflection_cooldown
        self.recent_sentences = []
        self.stop_injecting = False  # Flag to stop wait injection after answer



        # Hook & schedule state
        if "intervene_functions" in kwargs:
            self.intervene_functions = kwargs.pop("intervene_functions")  # pop to prevent serialization issues
            
            # Set the module for each intervention function
            layer_idx = kwargs.pop("layer_idx", 27)  # Default to layer 27 if not specified, pop to prevent serialization
            for intervene_function in self.intervene_functions:
                # intervene_function.module = self.model.layers[layer_idx].mlp.act_fn
                modules = [self.model.layers[i].mlp.act_fn for i in range(28)]
                intervene_function.modules = modules
    
    
    def _get_initial_cache_position(self, input_ids, model_kwargs):
        if hasattr(self, "_use_cache") and self._use_cache is False:
            return model_kwargs
        if "use_cache" not in model_kwargs:
            model_kwargs["use_cache"] = True
        if "cache_position" not in model_kwargs:
            model_kwargs["cache_position"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        return model_kwargs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Override from_pretrained to handle intervention functions and cyclical scheduling properly"""

        # Extract intervene + scheduling params first
        intervene_functions = kwargs.pop("intervene_functions", None)
        layer_idx = kwargs.pop("layer_idx", 27)

        # Extract cyclical parameters safely
        cyclical_params = {
            "wait_cyclical_amplitude": kwargs.pop("wait_cyclical_amplitude", 3.0),
            "wait_cyclical_period": kwargs.pop("wait_cyclical_period", 100.0),
            "wait_cyclical_shift": kwargs.pop("wait_cyclical_shift", 0.0),
        }

        # Load model weights using base class
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Assign intervention hooks
        if intervene_functions is not None:
            model.intervene_functions = intervene_functions
            modules = [model.model.layers[i].mlp.act_fn for i in range(28)]
            for intervene_function in intervene_functions:
                intervene_function.modules = modules

        # Assign cyclical schedule parameters
        model.wait_cyclical_amplitude = cyclical_params["wait_cyclical_amplitude"]
        model.wait_cyclical_period = cyclical_params["wait_cyclical_period"]
        model.wait_cyclical_shift = cyclical_params["wait_cyclical_shift"]

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
            slice_ids = token_ids[self.last_period_idx : cur_idx + 1]
            text_slice = self.tokenizer.decode(slice_ids, skip_special_tokens=True).strip()
            digit_count = sum(ch.isdigit() for ch in text_slice)

            # Avoid injecting on repetitive sentences
            if text_slice in self.recent_sentences:
                trigger = False
            elif self.sentences_since_reflection < self.reflection_cooldown:
                trigger = False
            elif digit_count >= min_digits and len(text_slice.split()) > 6:
                trigger = True
                self.sentences_since_reflection = 0
                self.recent_sentences.append(text_slice)
                self.recent_sentences = self.recent_sentences[-3:]
            else:
                trigger = False

            self.sentences_since_reflection += 1
            self.last_period_idx = cur_idx + 1
            return trigger, self.last_period_idx

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
        model_kwargs = dict(model_kwargs)

        # For cyclical wait injection
        self.recent_sentences = []
        self.last_injected_idx = 0
        current_step = 0
        wait_insertions = 0
        max_wait_tokens = 5  # Cap to avoid overwhelming output

        amplitude = getattr(self, "wait_cyclical_amplitude", 3.0)
        period = getattr(self, "wait_cyclical_period", 100.0)
        shift = getattr(self, "wait_cyclical_shift", 0.0)
        wait_token_ids = self.tokenizer.convert_tokens_to_ids(["wait", "Wait"])
        last_injected_step = -1000  # arbitrary large gap
        wait_injection_interval = 20  # only inject every N steps
        # Dynamic wait interval: starts high, decreases with time
        min_interval = 6
        max_interval = 30
        decay_speed = 200  # higher = slower decay

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
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            current_step += 1  # <-- Add this line to ensure correct wait timing

            #print("Step", current_step, "→ Generated token:", self.tokenizer.decode(next_tokens[0].item()))


# ==== Aggressively Inject "Wait" for Self-Reflection ====
            shifted_pos = (current_step + self.wait_cyclical_shift * self.wait_cyclical_period) % self.wait_cyclical_period
            cycle_pos = shifted_pos / self.wait_cyclical_period

            # Cosine-shaped injection window
            if cycle_pos <= 0.25:
                penalty = (cycle_pos / 0.25) * self.wait_cyclical_amplitude
            elif cycle_pos <= 0.75:
                penalty = self.wait_cyclical_amplitude - ((cycle_pos - 0.25) / 0.5) * 2 * self.wait_cyclical_amplitude
            else:
                penalty = -self.wait_cyclical_amplitude + ((cycle_pos - 0.75) / 0.25) * self.wait_cyclical_amplitude


            do_insert = True  # Always allow injection if should_inject is met
            cur_idx = input_ids.shape[1]
            recent_segment = self.tokenizer.decode(input_ids[0, self.last_injected_idx+1:].tolist())

            # Detect sentence boundaries or reflective cues
            sentence_boundary_like = any(phrase in recent_segment.lower() for phrase in ["so", "thus", "therefore", "now", "next", "let's"])
            is_eos = self._is_end_of_sentence(next_tokens[0].item()) or sentence_boundary_like

            # Define when we should inject "Wait"
            clean_segment = recent_segment.strip().lower()
            should_inject = (
                4 <= len(clean_segment.split()) <= 60 and
                clean_segment not in self.recent_sentences and
                (
                    sum(c.isdigit() for c in clean_segment) >= 1 or
                    any(trigger in clean_segment for trigger in ["let", "so", "then", "next", "consider", "step", "suppose"])
                )
            )

            # Optional debug logging
            # if not should_inject:
            #     print(f"[Skip] Not injecting at step {current_step}. Segment: {repr(recent_segment[-50:])}")

            generation_ratio = current_step / generation_config.max_new_tokens

            def injection_prob_curve(r: float) -> float:
                if r < 0.2:
                    return 0.1
                elif r < 0.4:
                    return 0.3 + 1.0 * (r - 0.2) / 0.2  # ramps up to 1.3
                elif r < 0.7:
                    return 1.3  # peak zone
                elif r < 0.9:
                    return 1.3 - 0.8 * (r - 0.7) / 0.2  # tapers to 0.5
                else:
                    return 0.3  # low again

            injection_prob = min(1.0, max(0.0, injection_prob_curve(generation_ratio)))
            
            # Adaptive interval: low at the start, tighter in mid/end
            if generation_ratio < 0.2:
                wait_injection_interval = 10
            elif generation_ratio < 0.5:
                wait_injection_interval = 6
            else:
                wait_injection_interval = 3

            # if generation_ratio < 0.2:
            #     wait_injection_interval = 6  # low frequency at the beginning
            # elif generation_ratio < 0.5:
            #     wait_injection_interval = 3  # increase during reasoning
            # else:
            #     wait_injection_interval = 2  # high frequency near the end
            
            if generation_ratio < 0.1:
                wait_injection_interval = 8
            elif generation_ratio < 0.3:
                wait_injection_interval = 5
            elif generation_ratio < 0.6:
                wait_injection_interval = 3
            else:
                wait_injection_interval = 2
            
            wait_injection_interval = int(max_interval - (max_interval - min_interval) * (1 - math.exp(-current_step / decay_speed)))
            # if not should_inject:
            #     print(f"[Skip] Not injecting at step {current_step}. Segment: {repr(recent_segment[-50:])}")
            if (
                not self.stop_injecting and
                wait_insertions < max_wait_tokens and
                should_inject and
                do_insert and
                (current_step - last_injected_step >= wait_injection_interval)
            ):
                if random.random() < injection_prob:
                    last_injected_step = current_step
                    self.last_injected_idx = input_ids.shape[1]

                    self.recent_sentences.append(recent_segment)
                    if len(self.recent_sentences) > 5:
                        self.recent_sentences.pop(0)

                    extra_ids = self.tokenizer("Wait", add_special_tokens=False).input_ids
                    extra_tensor = torch.tensor(extra_ids, device=input_ids.device).unsqueeze(0)
                    # print(f"[Inject] Step {current_step} → Injecting 'Wait' after: {repr(recent_segment[-40:])}")
                    
                    input_ids = torch.cat([input_ids, extra_tensor], dim=1)
                    wait_insertions += 1
                    clean_segment = recent_segment.strip().lower()
                    if clean_segment not in self.recent_sentences:
                        self.recent_sentences.append(clean_segment)
                        if len(self.recent_sentences) > 5:
                            self.recent_sentences.pop(0)
                            
                    # Forward pass for new token
                    forced_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    if output_attentions: forced_inputs["output_attentions"] = True
                    if output_hidden_states: forced_inputs["output_hidden_states"] = True
                    forced_out = model_forward(**forced_inputs, return_dict=True)

                    forced_logits = forced_out.logits[:, -1, :].float()
                    forced_scores = logits_processor(input_ids, forced_logits)

                    if output_scores: scores += (forced_scores,)
                    if output_logits: raw_logits += (forced_logits,)
                    if output_attentions:
                        decoder_attentions += (forced_out.attentions if not self.config.is_encoder_decoder else forced_out.decoder_attentions,)
                    if output_hidden_states:
                        decoder_hidden_states += (forced_out.hidden_states if not self.config.is_encoder_decoder else forced_out.decoder_hidden_states,)

                    model_kwargs = self._update_model_kwargs_for_generation(
                        forced_out, model_kwargs,
                        is_encoder_decoder=self.config.is_encoder_decoder
                    )

                    current_step += 1
                    continue
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
