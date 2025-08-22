#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 11:53:06 2025

@author: umbertocappellazzo
"""

import torch 
import torch.nn as nn
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer, LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings
from typing import Optional, Tuple, List, Union
import math
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


from transformers.utils import logging

logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "LlamaConfig"

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""



class Bottleneck_adapter(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        
        self.linear_downsample = nn.Linear(in_dim, bottleneck_dim)
        self.linear_upsample = nn.Linear(bottleneck_dim, in_dim)
        self.act = torch.nn.GELU()
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_downsample.weight) 
        nn.init.zeros_(self.linear_upsample.weight)
        nn.init.zeros_(self.linear_downsample.bias) 
        nn.init.zeros_(self.linear_upsample.bias)
    
    def forward(self, x):
        down_x = self.linear_downsample(x)
        up_x = self.linear_upsample(self.act(down_x))
        
        return up_x
    
class Hydra_Bottleneck_adapter(nn.Module):
    def __init__(self, bottleneck_dim, in_dim):
        super().__init__()
        
        self.linear = nn.Linear(bottleneck_dim, in_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.linear.weight) 
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x_output = self.linear(x)
        
        return x_output

@dataclass
class MoEAdapter_config:
    RANK: int
    N_EXPERTS: int  # The number of Matryoshka adapters.  
    TOP_K: int
    LOCATION: str # ["FFN", "MHSA", "LAYER"]
    NUM_SHARED_EXPERTS: int
    APPLY_LOAD_BAL_LOSS: bool
    IS_HYDRA: bool
    IS_TASK_SPECIFIC: bool = None
    

class LlamaForCausalLM_MoEAdapter(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: LlamaConfig, moeadapter_config: MoEAdapter_config):
        super().__init__(config)
        self.moeadapter_config= moeadapter_config
        self.model = LlamaModel_MoEAdapter(config, moeadapter_config)
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    #@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = None,
        modality = None
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs= self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            output_router_logits=output_router_logits,
            modality = modality
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        modality = None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "modality": modality
            }
        )
        return model_inputs


class LlamaModel_MoEAdapter(LlamaModel):
    def __init__(self, config: LlamaConfig, moeadapter_config: MoEAdapter_config):
        super().__init__(config)
        self.moeadapter_config= moeadapter_config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_MoEAdapter(config, layer_idx, moeadapter_config) for layer_idx in range(config.num_hidden_layers)]
        )
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_router_logits: Optional[bool] = None,
        modality = None
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        if output_router_logits:
            all_router_logits = ()

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_router_logits=output_router_logits,
                    modality = modality
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits if output_router_logits else None
        )
    

class Top_K_MoEAdapter(nn.Module):
    def __init__(self, num_experts, top_k, num_shared_experts, in_dim, bottleneck_dim, is_hydra, is_task_specific):
        super().__init__()
        
        
        if is_task_specific:
            self.gate = nn.ModuleDict({"audio": nn.Linear(in_dim, num_experts, bias=False),
                                       "video": nn.Linear(in_dim, num_experts, bias=False),
                                       "audiovisual": nn.Linear(in_dim, num_experts, bias=False)
                })
            if num_shared_experts == 0:
                self.shared_experts = None
            else:
                self.shared_experts = nn.ModuleDict({"audio": nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_shared_experts)]),
                                                     "video": nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_shared_experts)]),
                                                     "audiovisual": nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_shared_experts)])
                    })
            if not is_hydra:
                self.experts = nn.ModuleDict({"audio": nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_experts)]),
                                              "video": nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_experts)]),
                                              "audiovisual": nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_experts)])
                    })
            else:
                self.act = nn.ModuleDict({"audio": torch.nn.GELU(),
                                          "video": torch.nn.GELU(),
                                          "audiovisual": torch.nn.GELU()
                    })
                self.shared_downsampler = nn.ModuleDict({"audio": nn.Linear(in_dim, bottleneck_dim),
                                                         "video": nn.Linear(in_dim, bottleneck_dim),
                                                         "audiovisual": nn.Linear(in_dim, bottleneck_dim)
                    })
                nn.init.xavier_uniform_(self.shared_downsampler["audio"].weight); nn.init.zeros_(self.shared_downsampler["audio"].bias)
                nn.init.xavier_uniform_(self.shared_downsampler["video"].weight); nn.init.zeros_(self.shared_downsampler["video"].bias)
                nn.init.xavier_uniform_(self.shared_downsampler["audiovisual"].weight); nn.init.zeros_(self.shared_downsampler["audiovisual"].bias)
                self.experts = nn.ModuleDict({"audio": nn.ModuleList([Hydra_Bottleneck_adapter(bottleneck_dim, in_dim) for _ in range(num_experts)]),
                                              "video": nn.ModuleList([Hydra_Bottleneck_adapter(bottleneck_dim, in_dim) for _ in range(num_experts)]),
                                              "audiovisual": nn.ModuleList([Hydra_Bottleneck_adapter(bottleneck_dim, in_dim) for _ in range(num_experts)])
                    })
        else:
            self.gate = nn.Linear(in_dim, num_experts, bias=False)
            
            if num_shared_experts == 0:
                self.shared_experts = None
            else:
                self.shared_experts = nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_shared_experts)])
            
            if not is_hydra:
                self.experts = nn.ModuleList([Bottleneck_adapter(in_dim, bottleneck_dim) for _ in range(num_experts)])
            else:
                self.act = torch.nn.GELU()
                self.shared_downsampler = nn.Linear(in_dim, bottleneck_dim)
                nn.init.xavier_uniform_(self.shared_downsampler.weight); nn.init.zeros_(self.shared_downsampler.bias)
                self.experts = nn.ModuleList([Hydra_Bottleneck_adapter(bottleneck_dim, in_dim) for _ in range(num_experts)])
        
        self.top_k = top_k
        self.num_experts = num_experts
        self.is_hydra = is_hydra
        self.is_task_specific = is_task_specific
        
    def forward(self, hidden_states, output_router_logits = None, modality = None):
        
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate[modality](hidden_states) if self.is_task_specific else self.gate(hidden_states) 

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        if self.is_hydra:
            hidden_states = self.shared_downsampler[modality](self.act[modality](hidden_states)) if self.is_task_specific else self.shared_downsampler(self.act(hidden_states))
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[modality][expert_idx] if self.is_task_specific else self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        if self.shared_experts:
            if self.is_task_specific:
                for exp in self.shared_experts[modality]:
                    final_hidden_states += exp(hidden_states.view(batch_size, sequence_length, hidden_dim))
            else:
                for exp in self.shared_experts:
                    final_hidden_states += exp(hidden_states.view(batch_size, sequence_length, hidden_dim))
        
        if output_router_logits:
            return final_hidden_states, router_logits
        else:
            return final_hidden_states, None


class LlamaDecoderLayer_MoEAdapter(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, moeadapter_config: MoEAdapter_config):
        super().__init__(config, layer_idx)
        self.moeadapter_config = moeadapter_config
        
        self.adapter = Top_K_MoEAdapter(moeadapter_config.N_EXPERTS, moeadapter_config.TOP_K, moeadapter_config.NUM_SHARED_EXPERTS, 
                                        config.hidden_size, round(config.hidden_size/moeadapter_config.RANK), moeadapter_config.IS_HYDRA, 
                                        moeadapter_config.IS_TASK_SPECIFIC)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        output_router_logits = None,
        modality = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        
        if self.moeadapter_config.LOCATION == "MHSA":
            moeadapter_output, router_logits = self.adapter(hidden_states, output_router_logits = output_router_logits, modality = modality)
            
            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            
            hidden_states = residual + hidden_states + moeadapter_output
            
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            
            output_mlp = self.mlp(hidden_states)
            
            hidden_states = residual + output_mlp
        
        elif self.moeadapter_config.LOCATION == "FFN":  
        
            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            
            output_mlp = self.mlp(hidden_states)
            
            moeadapter_output, router_logits = self.adapter(hidden_states, output_router_logits = output_router_logits, modality = modality)
           
            hidden_states = residual + output_mlp + moeadapter_output
        
        else:
            moeadapter_output, router_logits = self.adapter(hidden_states, output_router_logits = output_router_logits, modality = modality)
            
            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            
            hidden_states = residual + hidden_states
            
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            
            output_mlp = self.mlp(hidden_states)
            
            hidden_states = residual + output_mlp + moeadapter_output
            
            
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        
        outputs += (router_logits,)

        return outputs
    
