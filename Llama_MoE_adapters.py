#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:51:22 2024

@author: umbertocappellazzo
"""

import torch 
import torch.nn as nn
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple
import warnings
import torch.nn.functional as F


# BOTTLENECK adapter module.

class Bottleneck_adapter(nn.Module):
    def __init__(self, in_dim, reduction_rate, out_dim):
        super().__init__()
        
        bottleneck_dim = round(in_dim/reduction_rate)
        self.linear_downsample = nn.Linear(in_dim, bottleneck_dim)
        self.linear_upsample = nn.Linear(bottleneck_dim, out_dim)
        #self.layer_norm_adapt = nn.LayerNorm(out_dim)  # If we want to add a LayerNorm after the up-projection layer.
        self.act = torch.nn.GELU()
        
        nn.init.zeros_(self.linear_downsample.weight); nn.init.zeros_(self.linear_upsample.weight)
        nn.init.zeros_(self.linear_downsample.bias); nn.init.zeros_(self.linear_upsample.bias);
        
    def forward(self, x):
        down_x = self.linear_downsample(x)
        up_x = self.linear_upsample(self.act(down_x))
        
        return up_x
        #return self.layer_norm_adapt(up_x)
   
# Sparse-MoA.

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

@dataclass
class SMoA_config_llama:
    REDUCTION_RATE: int 
    ADAPTER_LOCATION: str # 'MHSA' or 'FFN' or 'MHSA-FFN'
    NUM_EXPERTS: int
    TOP_K: int


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, reduction_rate_adapter):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Bottleneck_adapter(n_embed, reduction_rate_adapter, n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class LlamaForCausalLM_SMoA(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: LlamaConfig, SMoA_config: SMoA_config_llama):
        super().__init__(config)
        self.SMoA_config= SMoA_config
        self.model = LlamaModel_SMoA(config, SMoA_config)

class LlamaModel_SMoA(LlamaModel):
    def __init__(self, config: LlamaConfig, SMoA_config: SMoA_config_llama):
        super().__init__(config)
        self.SMoA_config= SMoA_config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_SMoA(config, layer_idx, SMoA_config) for layer_idx in range(config.num_hidden_layers)]
        )

class LlamaDecoderLayer_SMoA(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, SMoA_config: SMoA_config_llama):
        super().__init__(config, layer_idx)
        self.SMoA_config = SMoA_config
        self.config = config
        if self.SMoA_config.ADAPTER_LOCATION == "MHSA-FFN":
            self.SMoA_ffn = SparseMoE(self.config.hidden_size, self.SMoA_config.NUM_EXPERTS, self.SMoA_config.TOP_K, self.SMoA_config.REDUCTION_RATE)
            self.SMoA_mhsa = SparseMoE(self.config.hidden_size, self.SMoA_config.NUM_EXPERTS, self.SMoA_config.TOP_K, self.SMoA_config.REDUCTION_RATE)
        else:
            self.SMoA = SparseMoE(self.config.hidden_size, self.SMoA_config.NUM_EXPERTS, self.SMoA_config.TOP_K, self.SMoA_config.REDUCTION_RATE)
        #self.adapter_module = Bottleneck_adapter(config.hidden_size, self.adapter_config.REDUCTION_RATE, config.hidden_size)
    
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
        
        if self.SMoA_config.ADAPTER_LOCATION == 'MHSA':
            
            adapter_output = self.SMoA(hidden_states) 
            
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
            
            hidden_states = hidden_states + adapter_output + residual

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
        
        elif self.SMoA_config.ADAPTER_LOCATION == 'MHSA-FFN':
            adapter_output = self.SMoA_mhsa(hidden_states) 
            
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
            
            hidden_states = hidden_states + adapter_output + residual
            
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            
            output_mlp = self.mlp(hidden_states)
            adapter_output = self.SMoA_ffn(hidden_states) 
            hidden_states = residual + output_mlp + adapter_output
                
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs
            
        
        else: # FFN
            assert self.SMoA_config.ADAPTER_LOCATION == 'FFN'
        
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
            adapter_output = self.SMoA(hidden_states) 
            hidden_states = residual + output_mlp + adapter_output
                
            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs
        


# Dense MoA.

@dataclass
class DMoA_config_llama:
    REDUCTION_RATE: int 
    NUM_EXPERTS: int

class Router(nn.Module):
    def __init__(self, model_dim, num_adapters):#, is_soft_merging):
        super().__init__()
        
        #self.is_soft_merging = is_soft_merging
        self.model_dim, self.num_adapters = model_dim, num_adapters
        self.ff = nn.Linear(self.model_dim, self.num_adapters)
        
    def forward(self,x):
        # x shape: [B,L,H]  B = batch_size, L = sequence_length, H = hidden_size
        
        adapter_logits = self.ff(x)  # adapter_logits: [B,L,N], N = number_of_adapters
        adapter_probs = F.softmax(adapter_logits, dim=-1)   
        
        return adapter_probs     #,adapter_logits
    

class MoA(nn.Module):
    def __init__(self, n_embed, num_experts, reduction_rate):
        super().__init__()
        self.router = Router(n_embed, num_experts)
        self.moa = nn.ModuleList([Bottleneck_adapter(n_embed, reduction_rate, n_embed) for _ in range(num_experts)])
        
    
    def forward(self, x):
        # x shape: [B,L,H]  B = batch_size, L = sequence_length, H = hidden_size
        router_probs = self.router(x)
        output = torch.stack([adapter_module(x) for adapter_module in self.moa], dim=-1)
        output = (router_probs[:,:,None,:]*output).sum(-1)
        
        return output

        
    
class LlamaForCausalLM_DMoA(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: LlamaConfig, DMoA_config: DMoA_config_llama):
        super().__init__(config)
        self.DMoA_config= DMoA_config
        self.model = LlamaModel_DMoA(config, DMoA_config)

class LlamaModel_DMoA(LlamaModel):
    def __init__(self, config: LlamaConfig, DMoA_config: DMoA_config_llama):
        super().__init__(config)
        self.DMoA_config= DMoA_config
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_DMoA(config, layer_idx, DMoA_config) for layer_idx in range(config.num_hidden_layers)]
        )

class LlamaDecoderLayer_DMoA(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, DMoA_config: DMoA_config_llama):
        super().__init__(config, layer_idx)
        self.DMoA_config = DMoA_config
        self.config = config
        self.DMoA = MoA(self.config.hidden_size, self.DMoA_config.NUM_EXPERTS, self.DMoA_config.REDUCTION_RATE)
    
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
        
        dmoa_output = self.DMoA(hidden_states)
        
        hidden_states = residual + output_mlp + dmoa_output
            
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs