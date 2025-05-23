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
from typing import Optional, Union
# sampling 相关的工具
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

class BoxCountStoppingCriteria(StoppingCriteria):
    """
    Stops generation 50 tokens after the first box in generated text is detected.
    """
    def __init__(self, prompt_len: int, tokenizer, max_tokens_after_box: int = 100):
        self.prompt_len = prompt_len  # Length of the prompt to ignore
        self.box_detected = False
        self.tokens_after_box = 0
        self.max_tokens_after_box = max_tokens_after_box
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only check generated tokens (skip prompt)
        if input_ids.shape[1] <= self.prompt_len:
            return False
            
        # If we already detected a box, count tokens after it
        if self.box_detected:
            self.tokens_after_box += 1
            return self.tokens_after_box >= self.max_tokens_after_box
            
        # Check only the new token (last one) in the generated part
        if input_ids.shape[1] > self.prompt_len:
            # Get the last generated token
            last_token = input_ids[0, -1].item()
            last_token_str = self.tokenizer.decode([last_token])
            
            # Check if this token contains the box marker
            if "boxed" in last_token_str:
                self.box_detected = True
                
        return False

class DeepSeekQwenModel(Qwen2ForCausalLM):
    def __init__(
        self,
        # ref_neuron_list,
        # acc_neuron_list,
        # cold=0,
        # ref_count = 1,
        # acc_count = 1,
        # model_name="Qwen/Qwen2.5-Math-7B",
        config,
        **kwargs,
    ):
        super().__init__(config)

        # Hook & schedule state
        self.amplification_handle = None
        if "ref_neuron_list" in kwargs:
            self.ref_neuron_list = kwargs["ref_neuron_list"]
        else:
            raise ValueError("ref_neuron_list is required")
        if "acc_neuron_list" in kwargs:
            self.acc_neuron_list = kwargs["acc_neuron_list"]
        else:
            raise ValueError("acc_neuron_list is required")
        if "cold" in kwargs:
            self.cold = kwargs["cold"]
        else:
            raise ValueError("cold is required")
        if "ref_count" in kwargs:
            self.ref_count = kwargs["ref_count"]
        else:
            raise ValueError("ref_count is required")
        if "acc_amp" in kwargs:
            self.acc_amp = kwargs["acc_amp"]
        else:
            raise ValueError("acc_amp is required")
        if "acc_table" in kwargs:
            self.acc_table = set(kwargs["acc_table"])
        else:
            raise ValueError("acc_table is required")
        if "len_amp" in kwargs:
            self.len_amp = kwargs["len_amp"]
        else:
            raise ValueError("len_amp is required")
        if "ref_amp" in kwargs:
            self.ref_amp = kwargs["ref_amp"]
        self.t = 1
        self.current_sentence = []
        self.schedule_current = False
        self.refractory = 0
        self.acc_handle = None
        self.length_handle = None
        self.tokenizer = None  # Will be set during generation

    def register_length_hook(
            self,
            layer_idx: int,
            top_neurons: list[int],
            T_len: int
    ):
        self.token_position = 0
        try:
            layer = self.model.layers[layer_idx]
            act_module = layer.mlp.act_fn
        except Exception as e:
            raise ValueError(f"Couldn't locate layer {layer_idx}: {e}")

        def _length_hook(module, inputs, output):
            output[..., top_neurons] *= self.len_amp
            return output

        self.length_handle = act_module.register_forward_hook(_length_hook)
        # print(f"[DeepSeek] Registered length‐hook (add f) on layer {layer_idx}, T_len={T_len}")

    
    def remove_length_hook(self):
        if self.length_handle is not None:
            self.length_handle.remove()
            self.length_handle = None
            # print("[DeepSeek] Length hook removed.")


    def register_acc_hook(
            self,
            layer_idx: int,
            top_neurons: list[int]
    ):
        try:
            layer = self.model.layers[layer_idx]
            act_module = layer.mlp.act_fn
        except Exception as e:
            raise ValueError(f"Couldn't locate layer {layer_idx}: {e}")

        def _acc_hook(module, inputs, output):
            output[..., top_neurons] *= self.acc_amp
            return output
        
        self.remove_acc_hook()
        self.acc_handle = act_module.register_forward_hook(_acc_hook)
        # print(f"[DeepSeek] Registered acc‐hook (time {factor}) on layer {layer_idx}")


    def remove_acc_hook(self):
        if self.acc_handle is not None:
            self.acc_handle.remove()
            self.acc_handle = None
    

    def register_ref_hook(self, layer_idx: int, top_neurons: dict):
        layer = self.model.layers[layer_idx]
        act_fn = layer.mlp.act_fn
        top_neurons = list(top_neurons.keys())
        def _ref_hook(module, inputs, output):
            output[..., top_neurons] *= self.ref_amp
            return output
        self.remove_ref_hook()
        self.amplification_handle = act_fn.register_forward_hook(_ref_hook)


    def remove_ref_hook(self):
        if self.amplification_handle is not None:
            self.amplification_handle.remove()
            self.amplification_handle = None

    def _schedule_factor(self, t: int) -> float:
        a, b, c = 0.3170, 0.030, -0.9997
        denom = t + c
        if denom <= 0:
            denom = 1e-3
        return a - b * math.log(denom)
    

    def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
        return False


    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: BaseStreamer = None,
        acc_active=False,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        stopping_criteria = StoppingCriteriaList(
        crit for crit in stopping_criteria if not hasattr(crit, "eos_token_id")
        )
        
        # Add box stopping criteria
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            box_stopping = BoxCountStoppingCriteria(
                prompt_len=input_ids.shape[1],
                tokenizer=self.tokenizer
            )
            stopping_criteria.append(box_stopping)
            
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
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        self.register_acc_hook(layer_idx=27,top_neurons=self.acc_neuron_list)
        self.register_ref_hook(layer_idx=27, top_neurons=self.ref_neuron_list)

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

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
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