#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:14:56 2024

@author: umbertocappellazzo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:28:43 2024

@author: umbertocappellazzo
"""

import torch 
import torch.nn as nn
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer, LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
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


@dataclass
class MatryLoRA_config:
    TYPE: str
    RANK: int
    ALPHA: int
    N_LORA: int  # The number of Matryoshka LoRAs. 
    IS_LLAMA3: bool = False
    IS_LLAMA3_2_3B: bool = False
    

class LoRA(nn.Module):
    def __init__(self, rank, input_dim, bottleneck_dim, output_dim_V):
        super().__init__()
        
        self.lora_down_Q = nn.Linear(input_dim, bottleneck_dim, bias= False)
        self.lora_down_V = nn.Linear(input_dim, bottleneck_dim, bias= False)
        self.lora_up_Q = nn.Linear(bottleneck_dim, input_dim, bias= False)
        self.lora_up_V = nn.Linear(bottleneck_dim, output_dim_V, bias= False)
        
        nn.init.zeros_(self.lora_down_Q.weight)
        nn.init.kaiming_uniform_(self.lora_up_Q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down_V.weight)
        nn.init.kaiming_uniform_(self.lora_up_V.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        return self.lora_up_Q(self.lora_down_Q(hidden_states)), self.lora_up_V(self.lora_down_V(hidden_states))


class LlamaSdpaAttention_MatryLoRA(LlamaSdpaAttention):
    """
    LlamaSdpaAttention_MatryLoRA comprises multiple local LoRA modules and one global LoRA module.
    Local LoRAs are responsible to learn representations with specific compression rates.
    The Global LoRA processes all the representations associated with different compression rates. 
    
    During training and evaluation stages, all the local LoRAs are activated in order to learn representations with multiple granularities (i.e, Matryoshka representation learning). 
    During inference, we activate only one local LoRA based on the desired compression rate, identified by the matrylora_index, + the global LoRA.
    Note that the Global LoRA module is always activated during both training/val and inference stages.
    """
    
    
    def __init__(self, config: LlamaConfig, matrylora_config: MatryLoRA_config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        self.rank = matrylora_config.RANK
        self.scaling = matrylora_config.ALPHA/self.rank
        self.n_lora = matrylora_config.N_LORA
        self.matry_type = matrylora_config.TYPE
        
        hid_size = config.hidden_size
        
        # Need to take care of grouped-query attention. Different Llama models have different grouped-query attention size.
        if matrylora_config.IS_LLAMA3:
            output_dim_V = hid_size//4
        elif matrylora_config.IS_LLAMA3_2_3B:
            output_dim_V = hid_size//3
        else:
            output_dim_V =  hid_size
        
        self.local_loras = nn.ModuleList([LoRA(self.rank, hid_size, round(hid_size/self.rank), output_dim_V) for _ in range(self.n_lora)])
        if self.matry_type == "local-global":
            self.global_lora = LoRA(self.rank, hid_size, round(hid_size/self.rank), output_dim_V)
        
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    
    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        matrylora_index = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if output_attentions:
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        Q_lora_local, V_lora_local = self.local_loras[matrylora_index](hidden_states) 
        if self.matry_type == "local":
            query_states = query_states + Q_lora_local*self.scaling
            value_states = value_states + V_lora_local*self.scaling
        else:
            Q_lora_global, V_lora_global = self.global_lora(hidden_states) 
            query_states = query_states + (Q_lora_local + Q_lora_global)*self.scaling
            value_states = value_states + (V_lora_local + V_lora_global)*self.scaling 
        
        
        #key_states = key_states + K_lora*self.scaling 

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
        
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        
        #O_lora = self.lora_up_O(self.lora_down_O(hidden_states))
        
        attn_output = self.o_proj(attn_output) #+ O_lora*self.scaling

        return attn_output, None, past_key_value





# class ToyLlamaForCausalLM(LlamaForCausalLM):
    
#     _tied_weights_keys = ["lm_head.weight"]
    
#     def __init__(self, config: LlamaConfig, matrylora_config: MatryLoRA_config):
#         super().__init__(config)
#         self.matrylora_config= matrylora_config
#         self.model = ToyLlamaModel(config)
    
#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         matrylora_index = None,
#         ):
        
#         print("matrylora_index", matrylora_index)
#         return super().forward(input_ids,
#         attention_mask = None,
#         position_ids = None,
#         past_key_values = None,
#         inputs_embeds = None,
#         labels = None,
#         use_cache = None,
#         output_attentions = None,
#         output_hidden_states = None,
#         return_dict = None,
#         cache_position = None,
#         )
    
#     def prepare_inputs_for_generation(
#         self,
#         *args,
#         matrylora_index = None,
#         **kwargs,
#     ):
#         print("ciao")
#         output = super().prepare_inputs_for_generation(*args, **kwargs)
#         output.update({"matrylora_index": matrylora_index})
#         return output
        

class LlamaForCausalLM_MatryLoRA(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: LlamaConfig, matrylora_config: MatryLoRA_config):
        super().__init__(config)
        self.matrylora_config= matrylora_config
        self.model = LlamaModel_MatryLoRA(config, matrylora_config)
    
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
        matrylora_index = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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
            matrylora_index = matrylora_index
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
        matrylora_index = None,
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
                "matrylora_index": matrylora_index
            }
        )
        return model_inputs

        

class LlamaModel_MatryLoRA(LlamaModel):
    def __init__(self, config: LlamaConfig, matrylora_config: MatryLoRA_config):
        super().__init__(config)
        self.matrylora_config= matrylora_config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_MatryLoRA(config, layer_idx, matrylora_config) for layer_idx in range(config.num_hidden_layers)]
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
        matrylora_index = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        
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
                    matrylora_index = matrylora_index
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    
    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     cache_position=None,
    #     position_ids=None,
    #     use_cache=True,
    #     matrylora_index = None,
    #     **kwargs,
    # ):
    #     print("Ciao!!!!")
    #     # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    #     # Exception 1: when passing input_embeds, input_ids may be missing entries
    #     # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    #     if past_key_values is not None:
    #         if inputs_embeds is not None:  # Exception 1
    #             input_ids = input_ids[:, -cache_position.shape[0] :]
    #         elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
    #             input_ids = input_ids[:, cache_position]

    #     if attention_mask is not None and position_ids is None:
    #         # create position_ids on the fly for batch generation
    #         position_ids = attention_mask.long().cumsum(-1) - 1
    #         position_ids.masked_fill_(attention_mask == 0, 1)
    #         if past_key_values:
    #             position_ids = position_ids[:, -input_ids.shape[1] :]

    #     # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    #     if inputs_embeds is not None and cache_position[0] == 0:
    #         model_inputs = {"inputs_embeds": inputs_embeds}
    #     else:
    #         model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

    #     model_inputs.update(
    #         {
    #             "position_ids": position_ids,
    #             "cache_position": cache_position,
    #             "past_key_values": past_key_values,
    #             "use_cache": use_cache,
    #             "attention_mask": attention_mask,
    #             "matrylora_index": matrylora_index
    #         }
    #     )
    #     return model_inputs
    
    
    

class LlamaDecoderLayer_MatryLoRA(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx, matrylora_config: MatryLoRA_config):
        super().__init__(config, layer_idx)
        self.matrylora_config= matrylora_config
        
        self.self_attn = LlamaSdpaAttention_MatryLoRA(config=config, layer_idx=layer_idx, matrylora_config=matrylora_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        matrylora_index = None,
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
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            matrylora_index = matrylora_index,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



    

#if __name__ == "__main__":
    #toymodel = LlamaForCausalLM_MatryLoRA.from_pretrained("meta-llama/Llama-3.2-1B", MatryLoRA_config(32,4,5,True, False))
#    toymodel = ToyLlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", MatryLoRA_config(32,4,5,True, False))
#    toymodel.generate(matrylora_index=1, max_length=5)
