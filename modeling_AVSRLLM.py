#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:09:38 2024

@author: umbertocappellazzo
"""

import torch
from torch import nn
import torch.nn.functional as F
#import numpy as np

from espnet.nets.pytorch_backend.frontend.resnet import video_resnet
from espnet.nets.pytorch_backend.frontend.resnet1d import audio_resnet
from espnet.nets.pytorch_backend.encoder.conformer_encoder import ConformerEncoder
from Llama_adapters import LlamaForCausalLM_adapter 
from Llama_LoRA import LlamaForCausalLM_lora
from Llama_MatryLoRA import LlamaForCausalLM_MatryLoRA
from Llama_MatryAdapter import LlamaForCausalLM_MatryAdapter
from Llama_MatryMoEAdapter import LlamaForCausalLM_MatryMoEAdapter
from Llama_MoEAdapter import LlamaForCausalLM_MoEAdapter
from Llama_MatryMoELoRA import LlamaForCausalLM_MatryMoELoRA
from Mistral_LoRA import MistralForCausalLM_lora
from Qwen_LoRA import Qwen2ForCausalLM_lora
#from espnet.nets.pytorch_backend.encoder.whisper_encoder import OpenAIWhisperEncoder
#from espnet.nets.pytorch_backend.frontend.whisper import WhisperFrontend
from espnet.raven.nets.pytorch_backend.transformer.encoder import Encoder
from transformers import WhisperModel, LlamaForCausalLM, AutoFeatureExtractor, WavLMModel
from transformers import GemmaForCausalLM, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
#from dataclasses import dataclass
from peft import get_peft_model
from Attention import Attention
from Llama_MatryMoELoRA import Top_K_MoELoRA
#from sklearn.metrics.pairwise import cosine_similarity

import fairseq
from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq, AVHubertSeq2SeqConfig
from av_hubert.avhubert.hubert_lora import AVHubertModel_lora
import math

#from av_hubert.avhubert.hubert import AVHubertModel
#from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq
#from AV_HuBERT_encoder import avhubertConfig, avhubert_encoder

IGNORE_INDEX = -100

def load_balancing_loss_func(gate_logits, num_experts, top_k):
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if isinstance(gate_logits, tuple):
        gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0)
    
    routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class AVSR_LLMs(nn.Module):
    def __init__(self, modality, is_MoE, pretrain_avhubert_enc_video,  use_lora_avhubert, llm_model, hidden_size, 
                 intermediate_size, tokenizer, prompt, pad_id, 
                 downsample_ratio_audio, downsample_ratio_video, audio_encoder_name, 
                 unfrozen_modules, max_dec_tokens, num_beams, PETF_LLM_name = None, peft_config_llm = None, 
                 compression_mode = "stack", remove_layernorm_from_projector = False, is_matryoshka = False
                 ):
        
        super().__init__()
        
        self.modality = modality
        self.is_MoE = is_MoE
        self.pretrain_avhubert_enc_video = pretrain_avhubert_enc_video
        self.max_dec_tokens = max_dec_tokens
        self.num_beams = num_beams
        self.downsample_ratio_audio = downsample_ratio_audio
        self.downsample_ratio_video = downsample_ratio_video
        self.audio_encoder_name = audio_encoder_name
        self.llm_model = llm_model
        self.peft_config_llm = peft_config_llm
        self.PETF_LLM_name = PETF_LLM_name
        self.compression_mode = compression_mode
        self.hidden_size = hidden_size
        self.remove_layernorm_from_projector = remove_layernorm_from_projector
        self.is_matryoshka = is_matryoshka
            
        if modality == "audio" or modality == "audiovisual":
           
            if "whisper" in self.audio_encoder_name:
                
                print("Instantiating whisper!")    
                self.audio_encoder = WhisperModel.from_pretrained(self.audio_encoder_name).encoder
                self.audio_frontend = AutoFeatureExtractor.from_pretrained(self.audio_encoder_name)
                self.audio_encoder.requires_grad_(False)
                self.audio_encoder.train() # This must be explicitly done as by default the from_pretrained HF models are in eval mode when initialized (this is the opposite for pytorch!)--> cause a break in deepspeed 3! https://github.com/Lightning-AI/pytorch-lightning/issues/19467
                #self.audio_encoder.conv1.requires_grad_(True)
                #self.audio_encoder.conv2.requires_grad_(True)
                audio_dim =self.audio_encoder.config.hidden_size
                
            else: # WavLM.
                print("Instantiating WavLM!")    
                assert "wavlm" in self.audio_encoder_name, ("Only whisper and WavLM audio encoders are supported as of now!")
                self.audio_encoder = WavLMModel.from_pretrained(self.audio_encoder_name)
                self.audio_encoder.requires_grad_(False)
                audio_dim =self.audio_encoder.config.hidden_size
                
            
            if self.compression_mode == "stack":
                
                if self.is_matryoshka:
                    self.matry_map_audio = {}
                    for index, el in enumerate(self.downsample_ratio_audio):
                        self.matry_map_audio[el] = index
                
                if self.remove_layernorm_from_projector:
                    if self.is_matryoshka:
                        print("Instantiating stack projector for audio!")
                        self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for downsample_ratio in self.downsample_ratio_audio])
                    else:
                        self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for downsample_ratio in self.downsample_ratio_audio])
                else:
                    if self.is_matryoshka:
                        print("Instantiating stack projector for audio!")
                        self.audio_proj = nn.Sequential(nn.Linear(audio_dim*self.downsample_ratio_audio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
                    else:
                        self.audio_proj = nn.Sequential(nn.Linear(audio_dim*self.downsample_ratio_audio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
            elif self.compression_mode == "avg-pooling":
                if self.is_matryoshka:
                    self.matry_map_audio = {}
                    for index, el in enumerate(self.downsample_ratio_audio):
                        self.matry_map_audio[el] = index
                
                
                if self.is_matryoshka:
                    self.avg_pool_audio = nn.ModuleList([nn.AvgPool1d(downsample_ratio) for downsample_ratio in self.downsample_ratio_audio])
                else:
                    self.avg_pool_audio = nn.AvgPool1d(self.downsample_ratio_audio)
                
                if self.remove_layernorm_from_projector:
                    if self.is_matryoshka:
                        print("Instantiating avg-pooling projector for Matry audio!")
                        self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for _ in self.downsample_ratio_audio])
                    else:
                        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                else:
                    if self.is_matryoshka:
                        print("Instantiating avg-pooling projector for Matry audio!")
                        self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size)) for _ in self.downsample_ratio_audio])
                    else:
                        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
                
        if modality == "video" or modality == "audiovisual":
            if pretrain_avhubert_enc_video:
                print("Initializing AV-HuBERT Large, non fine-tuned!")
                
                if use_lora_avhubert:
                    
                    modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_video])
                    self.video_encoder = modell[0]
                    
                    print('Preparing LoRA layers for AV-HuBERT video-only!')
                    
                    
                    num_layers_avhubert = 24 if "large" in self.pretrain_avhubert_enc_video else 12
                    video_dim_avhubert = 1024 if "large" in self.pretrain_avhubert_enc_video else 768
                    for layer_idx in range(num_layers_avhubert):
                        # We set apply_lora = True for each video encoder layer such that it is applied. TODO: define this parameter in the AV-HuBERT main class.
                        self.video_encoder.encoder.layers[layer_idx].apply_lora = True
                        
                        self.video_encoder.encoder.layers[layer_idx].self_attn.rank = 16
                        self.video_encoder.encoder.layers[layer_idx].self_attn.scaling_lora = 2
                        
                        self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q = nn.Linear(video_dim_avhubert, round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), bias= False)
                        self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q = nn.Linear(round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), video_dim_avhubert, bias= False)
                        self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_V = nn.Linear(video_dim_avhubert, round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), bias= False)
                        self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_V = nn.Linear(round(video_dim_avhubert/self.video_encoder.encoder.layers[layer_idx].self_attn.rank), video_dim_avhubert, bias= False)
            
                        nn.init.zeros_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_Q.weight)
                        nn.init.zeros_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_down_V.weight)
                        nn.init.kaiming_uniform_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_Q.weight, a=math.sqrt(5))
                        nn.init.kaiming_uniform_(self.video_encoder.encoder.layers[layer_idx].self_attn.lora_up_V.weight, a=math.sqrt(5))
                    
                else:
                    modell, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.pretrain_avhubert_enc_video])
                    self.video_encoder = modell[0]
                    
                self.video_encoder.requires_grad_(False)
                video_dim = 1024 if "large" in self.pretrain_avhubert_enc_video else 768
                
                
            if self.compression_mode == "stack":
                if self.is_matryoshka:
                    self.matry_map_video = {}
                    for index, el in enumerate(self.downsample_ratio_video):
                        self.matry_map_video[el] = index
                
                
                if self.remove_layernorm_from_projector:
                    if self.is_matryoshka:
                        print("Instantiating stack projector for Matry video!")
                        self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for downsample_ratio in self.downsample_ratio_video])
                    else:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim*self.downsample_ratio_video, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                else:
                    if self.is_matryoshka:
                        print("Instantiating stack projector for Matry video!")
                        self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size)) for downsample_ratio in self.downsample_ratio_video])
                    else:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim*self.downsample_ratio_video, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))

            elif self.compression_mode == "avg-pooling":
                if self.is_matryoshka:
                    self.matry_map_video = {}
                    for index, el in enumerate(self.downsample_ratio_video):
                        self.matry_map_video[el] = index
                
                
                if self.is_matryoshka:
                    self.avg_pool_video = nn.ModuleList([nn.AvgPool1d(downsample_ratio) for downsample_ratio in self.downsample_ratio_video])
                else:
                    self.avg_pool_video = nn.AvgPool1d(self.downsample_ratio_video)
                    
                if self.remove_layernorm_from_projector:
                    if self.is_matryoshka:
                        print("Instantiating avg-pooling projector for Matry video!")
                        self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for _ in self.downsample_ratio_video])
                    else:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                else:
                    if self.is_matryoshka:
                        print("Instantiating avg-pooling projector for Matry video!")
                        self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size)) for _ in self.downsample_ratio_video])
                    else:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
    
        if "llama" in llm_model or "Tiny" in llm_model:
            if self.PETF_LLM_name is None:
                self.llm = LlamaForCausalLM.from_pretrained(llm_model)
            elif self.PETF_LLM_name == "adapter":
                if self.is_MoE:
                    self.llm = LlamaForCausalLM_MoEAdapter.from_pretrained(llm_model, peft_config_llm)
                else:
                    self.llm = LlamaForCausalLM_adapter.from_pretrained(llm_model, peft_config_llm)
            elif self.PETF_LLM_name == "lora":
                self.llm = LlamaForCausalLM_lora.from_pretrained(llm_model, peft_config_llm)
            
            #self.llm.train()  # See above WavLM/CLIP. 
        elif "gemma" in llm_model:
            if self.PETF_LLM_name is None:
                self.llm = GemmaForCausalLM.from_pretrained(llm_model)
            else:
                self.llm = get_peft_model(LlamaForCausalLM.from_pretrained(llm_model), peft_config_llm)
                self.llm.print_trainable_parameters()
                print(self.llm.config)
        elif "mistral" in llm_model:
            if self.PETF_LLM_name is None:
                self.llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
            elif self.PETF_LLM_name == "lora":
                self.llm = MistralForCausalLM_lora.from_pretrained(llm_model, peft_config_llm)
            else:
                self.llm = get_peft_model(LlamaForCausalLM.from_pretrained(llm_model), peft_config_llm)
                self.llm.print_trainable_parameters()
                print(self.llm.config)
        elif "Qwen" in llm_model:
            if self.PETF_LLM_name == "lora":
                self.llm = Qwen2ForCausalLM_lora.from_pretrained(llm_model, peft_config_llm) 
        # IMPORTANT: we need to add the pad_id to the model and resize the token embeddings matrix accordingly.
        self.tokenizer = tokenizer
        if "llama" in llm_model or "Tiny" in llm_model or "mistral" in llm_model:
            self.llm.config.pad_token_id = pad_id
        
        self.llm.resize_token_embeddings(len(self.tokenizer))
        if self.PETF_LLM_name != "lora_peft":
            self.llm.requires_grad_(False)
        
        self.prompt = prompt
        
        self._unfreeze_PETF(unfrozen_modules)
        
        
    def _unfreeze_PETF(self, unfrozen_modules):
        """
        Modules to be unfrozen. Unfrozen_blocks is a list with one or multiple values: ['peft_audio','peft_video','embedding','peft_llm']. 
        """
        if None in unfrozen_modules:
            return
        if "embedding" in unfrozen_modules: # the embedding matrix is learnable. The additional parameters to update are: 32000*4096 ~ 13M.
            self.llm.model.embed_tokens.requires_grad_(True)
        if "peft_llm" in unfrozen_modules:
            if self.PETF_LLM_name == "adapter":
                for block_idx in range(self.llm.config.num_hidden_layers):
                        self.llm.model.layers[block_idx].adapter.requires_grad_(True)
            
            elif self.PETF_LLM_name == "lora":
                print("Unfreezing LoRA for LLM:")
                for block_idx in range(self.llm.config.num_hidden_layers):
                    self.llm.model.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
        
        if "lora_avhubert" in unfrozen_modules:
            
            if self.modality == "video": 
                print("Unfreezing LoRA for AV-HuBERT video encoder!")
                
                
                for block_idx in range(24):
                    # If we don't use the correct config parameters --> my initial implementation.
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                    self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
            
        
    def forward(self, inputs, is_trainval= True, test_ratio_matry = None):
        
        embeddings, labels = self.prepare_inputs(inputs, is_trainval, test_ratio_matry = test_ratio_matry)
        
        
        if is_trainval: # Train/eval step: compute the logits and loss. Note that the llm computes itself the loss since we feed labels.
            
        
            if self.is_matryoshka:
                matryoshka_loss = 0.
                
                for matryoshka_emb, matryoshka_label in zip(embeddings, labels):
                    matryoshka_loss += self.llm(inputs_embeds = matryoshka_emb, labels = matryoshka_label)[0]
                
                matryoshka_loss /= len(embeddings)
                return matryoshka_loss
            
            else:
                if self.is_MoE:
                    outputs = self.llm(inputs_embeds = embeddings, labels = labels, output_router_logits = True)
                    return outputs[0], load_balancing_loss_func(outputs.router_logits, self.peft_config_llm.N_EXPERTS, self.peft_config_llm.TOP_K)
                else:
                    outputs = self.llm(inputs_embeds = embeddings, labels = labels)
                    return outputs[0]
                
        
        else: # Inference step: we decode starting from the audio/video tokens + prompt tokens (if used) + bos. 
           
            if self.llm_model ==  "meta-llama/Meta-Llama-3-8B" or self.llm_model == "meta-llama/Meta-Llama-3.1-8B" or self.llm_model == "meta-llama/Llama-3.2-1B" or self.llm_model == "meta-llama/Llama-3.2-3B":
                decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<|end_of_text|>"], 
                                                    bos_token_id = self.tokenizer.vocab["<|begin_of_text|>"], 
                                                    pad_token_id = self.tokenizer.vocab["<pad>"],
                                                    )
            elif self.llm_model in  ["TinyLlama/TinyLlama_v1.1", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-hf"]: # Llama 2.
                decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["</s>"], 
                                                bos_token_id = self.tokenizer.vocab["<s>"], 
                                                pad_token_id = self.tokenizer.vocab["<pad>"],
                                                )
            elif self.llm_model in ["google/gemma-2b","google/gemma-7b", "google/gemma-2-9b"] :
                decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<eos>"], 
                                                bos_token_id = self.tokenizer.vocab["<bos>"], 
                                                pad_token_id = self.tokenizer.vocab["<pad>"],
                                                )
            elif self.llm_model == "mistralai/Mistral-7B-v0.1":
                decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["</s>"], 
                                                bos_token_id = self.tokenizer.vocab["<s>"], 
                                                pad_token_id = self.tokenizer.vocab["<pad>"],
                                                #repetition_penalty=1.5
                                                )
            elif self.llm_model in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]:
                decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<|endoftext|>"], 
                                                pad_token_id = self.tokenizer.vocab["<|endoftext|>"],
                                                )
            return decoded_ids
            
    
    def prepare_inputs(self, inputs, is_trainval, test_ratio_matry = None):

        audio_features = self.encode_audio(inputs["audio"], max(inputs["lengths"]), is_trainval, test_ratio_matry = test_ratio_matry) if self.modality in ["audio", "audiovisual"] else None
        video_features = self.encode_video(inputs["video"], is_trainval, test_ratio_matry = test_ratio_matry) if self.modality in ["video", "audiovisual"] else None
        
        
        text_embeddings_ = self.llm.model.embed_tokens(inputs["tokens"])
        
        ignore_count = 0 
        
        
        # An important note here: the tokenizer by default inserts the EOS and BOS tokens. Since we do that already in the collate_LLM, here we need to
        # get rid of them explicitly --> [:,1:-1].
        #prompt_ids = self.tokenizer(self.prompt, return_tensors = "pt").input_ids.to(text_embeddings_.device)
        prompt_tokens_start_at = 0 if "Qwen" in self.llm_model else 1
        prompt_ids = self.tokenizer(self.prompt, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1].to(text_embeddings_.device)
        prompt_embeddings = self.llm.model.embed_tokens(prompt_ids.expand(inputs["tokens"].shape[0],-1))
        #print("Prompt tokens: ", prompt_ids[0,:])
        
        if is_trainval:
            if "Qwen" in self.llm_model:
                    text_embeddings = torch.cat([prompt_embeddings, text_embeddings_], dim=1)
            else:
                text_embeddings = torch.cat(
                    [torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1), text_embeddings_[:, 1:, :]], 
                    dim=1)
        else:
            #text_embeddings = prompt_embeddings
            if "Qwen" in self.llm_model:
                    text_embeddings = prompt_embeddings
            else:
                text_embeddings = torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_embeddings], dim=1)
        
        ignore_count += prompt_embeddings.shape[1]
        
        
        
        if self.is_matryoshka:
            if self.modality == "audiovisual":
                video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_starts =  self.llm.model.embed_tokens(video_starts)
                
                video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_ends = self.llm.model.embed_tokens(video_ends)
                
                audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_starts =  self.llm.model.embed_tokens(audio_starts)
                
                audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_ends = self.llm.model.embed_tokens(audio_ends)
                
                if is_trainval:
                    ignore_count = [ignore_count]*len(self.downsample_ratio_video)*len(self.downsample_ratio_audio)
                    audiovisual_matryoshka = []
                    counter = 0
                    for index_video,proj_video in enumerate(self.video_proj):
                        video_feat = proj_video(video_features[index_video])
                        video_inputs = torch.cat((video_starts, video_feat, video_ends), dim =1)
                        video_features_ = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), video_inputs, text_embeddings[:, 1:, :]), dim=1)
                        
                        for index_audio,proj_audio in enumerate(self.audio_proj):
                            audio_feat = proj_audio(audio_features[index_audio])
                            audio_inputs = torch.cat((audio_starts, audio_feat, audio_ends), dim =1)
                            
                            ignore_count[counter] += audio_inputs.shape[1] + video_inputs.shape[1]
                            audiovisual_matryoshka.append(torch.cat((video_features_[:, 0, :].unsqueeze(1), audio_inputs, video_features_[:, 1:, :]), dim=1))
                            counter += 1
                else:
                    video_features = self.video_proj[self.matry_map_video[test_ratio_matry[0]]](video_features)
                    video_inputs = torch.cat((video_starts, video_features, video_ends), dim = 1)
                    ignore_count += video_inputs.shape[1]
                    text_embeddings = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), video_inputs, text_embeddings[:, 1:, :]), dim = 1)
                    
                    audio_features = self.audio_proj[self.matry_map_audio[test_ratio_matry[1]]](audio_features)
                    audio_inputs = torch.cat((audio_starts, audio_features, audio_ends), dim = 1)
                    
                    ignore_count += audio_inputs.shape[1]
                    text_embeddings = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), audio_inputs, text_embeddings[:, 1:, :]), dim = 1)
                    
                if inputs["labels"] is not None:
                        labels = [torch.tensor([IGNORE_INDEX]*ignore_count_el, device=text_embeddings.device).expand(text_embeddings.shape[0], -1) for ignore_count_el in ignore_count]
                        labels = [torch.cat((inputs["labels"][:, 0].unsqueeze(1), label, inputs["labels"][:, 1:]), dim =1) for label in labels]
                else:
                    labels = None
                    
                
                if is_trainval:
                    return audiovisual_matryoshka, labels
                else:
                    return text_embeddings, labels
                
            else:
                    if video_features is not None:
                        video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                        video_starts = self.llm.base_model.model.model.embed_tokens(video_starts) if self.PETF_LLM_name == "lora_peft" else self.llm.model.embed_tokens(video_starts)
                        
                        video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                        video_ends = self.llm.base_model.model.model.embed_tokens(video_ends) if self.PETF_LLM_name == "lora_peft" else self.llm.model.embed_tokens(video_ends)
                    
                        if is_trainval:
                            ignore_count = [ignore_count]*len(self.downsample_ratio_video) if self.compression_mode in ["stack", "avg-pooling"] else [ignore_count]*len(self.resampler_num_tokens_video)
                            for index,proj in enumerate(self.video_proj):
                                video_feat = proj(video_features[index])
                                video_inputs = torch.cat((video_starts, video_feat, video_ends), dim =1)
                                ignore_count[index] += video_inputs.shape[1]
                                video_features[index] = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), video_inputs, text_embeddings[:, 1:, :]), dim=1)
                        else:
                            video_features = self.video_proj[self.matry_map_video[test_ratio_matry]](video_features)
                            video_inputs = torch.cat((video_starts, video_features, video_ends), dim = 1)
                            ignore_count += video_inputs.shape[1]
                            text_embeddings = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), video_inputs, text_embeddings[:, 1:, :]), dim = 1)
                
                    if audio_features is not None:
                        audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                        audio_starts =  self.llm.base_model.model.model.embed_tokens(audio_starts) if self.PETF_LLM_name == "lora_peft" else self.llm.model.embed_tokens(audio_starts)
                        
                        audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                        audio_ends = self.llm.base_model.model.model.embed_tokens(audio_ends) if self.PETF_LLM_name == "lora_peft" else self.llm.model.embed_tokens(audio_ends)
                        
                        if is_trainval:
                            if self.modality == "audio":
                                ignore_count = [ignore_count]*len(self.downsample_ratio_audio) if self.compression_mode in ["stack", "avg-pooling"] else [ignore_count]*len(self.resampler_num_tokens_audio)
                            for index,proj in enumerate(self.audio_proj):
                                audio_feat = proj(audio_features[index])
                                audio_inputs = torch.cat((audio_starts, audio_feat, audio_ends), dim =1)
                                ignore_count[index] += audio_inputs.shape[1]
                                audio_features[index] = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), audio_inputs, text_embeddings[:, 1:, :]), dim=1)
                        else:
                            audio_features = self.audio_proj[self.matry_map_audio[test_ratio_matry]](audio_features)
                            audio_inputs = torch.cat((audio_starts, audio_features, audio_ends), dim = 1)
                            ignore_count += audio_inputs.shape[1]
                            text_embeddings = torch.cat((text_embeddings[:, 0, :].unsqueeze(1), audio_inputs, text_embeddings[:, 1:, :]), dim = 1)
                    
                    if inputs["labels"] is not None:
                        labels = [torch.tensor([IGNORE_INDEX]*ignore_count_el, device=text_embeddings.device).expand(text_embeddings.shape[0], -1) for ignore_count_el in ignore_count]
                        labels = [torch.cat((inputs["labels"][:, 0].unsqueeze(1), label, inputs["labels"][:, 1:]), dim =1) for label in labels]
                    else:
                        labels = None
                    
                    if is_trainval:
                        if self.modality == "audio":
                            return audio_features, labels
                        else:
                            return video_features, labels
                    else:
                        return text_embeddings, labels
        
        else:
            if video_features is not None:
                video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_starts =  self.llm.model.embed_tokens(video_starts)
                
                video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_ends = self.llm.model.embed_tokens(video_ends)
                
                
                video_features = self.video_proj(video_features)
                
                #video_features = video_features.transpose(1,2)
                
                video_inputs = torch.cat([torch.cat([video_starts, video_features], dim=1), video_ends], dim=1)
               
                #***
                #counts.append(video_inputs.shape[1])
                
                if "Qwen" in self.llm_model:
                    text_embeddings = torch.cat([video_inputs, text_embeddings], dim=1)
                else:
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), video_inputs], dim=1), text_embeddings[:, 1:, :]], 
                        dim=1)
                    
                ignore_count += video_inputs.shape[1]
                
                #if is_trainval: ignore_count_video = video_inputs.shape[1] if is_trainval else None
            
            if audio_features is not None:
                audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_starts =  self.llm.model.embed_tokens(audio_starts)
                
                audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_ends = self.llm.model.embed_tokens(audio_ends)
                
                
                audio_features = self.audio_proj(audio_features)
                
                #audio_features = self.audio_proj_lin(audio_features)
                #audio_features= self.audio_proj_conv(audio_features.view(audio_features.shape[0], audio_features.shape[-1], -1))
                #audio_features = audio_features.view(audio_features.shape[0], audio_features.shape[-1], -1)
            
                audio_inputs = torch.cat([torch.cat([audio_starts, audio_features], dim=1), audio_ends], dim=1)
                
                #***
                #counts.append(audio_inputs.shape[1])
                if "Qwen" in self.llm_model:
                    text_embeddings = torch.cat([audio_inputs, text_embeddings], dim=1)
                else:
                    text_embeddings = torch.cat(
                        [torch.cat([text_embeddings[:, 0, :].unsqueeze(1), audio_inputs], dim=1), text_embeddings[:, 1:, :]], 
                        dim=1)
                    
                ignore_count += audio_inputs.shape[1]
                
                #print("Audio shape: : ", audio_inputs.shape)
    
            if inputs["labels"] is not None:
                labels = torch.tensor([IGNORE_INDEX]*ignore_count, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
                if "Qwen" in self.llm_model:
                    labels = torch.cat([labels, inputs["labels"]], dim=1)
                else:
                    labels = torch.cat(
                        [torch.cat([inputs["labels"][:, 0].unsqueeze(1), labels], dim=1), inputs["labels"][:, 1:]], 
                        dim=1)
            else:
                labels = None
            
            
            return text_embeddings, labels
        
    
    def encode_video(self, videos, is_trainval, test_ratio_matry = None):
        
        video_enc, _, encoder_layers = self.video_encoder.extract_finetune(source={'video': torch.reshape(videos,(-1,videos.shape[2],videos.shape[1],videos.shape[3],videos.shape[-1])),'audio': None})
        
        if self.is_matryoshka:
            if self.compression_mode == "stack":
                if is_trainval:
                    video_enc_list = []
                    for downsample_ratio in self.downsample_ratio_video:
                        video_temp = video_enc
                        video_enc_ = [video_temp[:, x:x + downsample_ratio, :].view(video_temp.shape[0], 1, -1) for x in range(0, video_temp.shape[1], downsample_ratio)]
                        rest = video_temp.shape[1] % downsample_ratio
                        if rest == 0:
                            video_enc_ = torch.stack(video_enc_, dim=1).squeeze(2) 
                        else: 
                            video_enc_ = torch.stack(video_enc_[:-1], dim=1).squeeze(2)
                        video_enc_list.append(video_enc_)
                    return video_enc_list
                else:
                    if test_ratio_matry != 1:
                        video_temp = video_enc
                        if self.modality == "audiovisual":
                            video_enc = [video_temp[:, x:x + test_ratio_matry[0], :].view(video_temp.shape[0], 1, -1) for x in range(0, video_temp.shape[1], test_ratio_matry[0])]
                            rest = video_temp.shape[1] % test_ratio_matry[0]
                        else:
                            video_enc = [video_temp[:, x:x + test_ratio_matry, :].view(video_temp.shape[0], 1, -1) for x in range(0, video_temp.shape[1], test_ratio_matry)]
                            rest = video_temp.shape[1] % test_ratio_matry
                        if rest == 0:
                            video_enc = torch.stack(video_enc, dim=1).squeeze(2) 
                        else: 
                            video_enc = torch.stack(video_enc[:-1], dim=1).squeeze(2)
                    return video_enc
        
            elif self.compression_mode == "avg-pooling":
                if is_trainval:
                    video_enc_list = []
                    video_enc = video_enc.transpose(1,2).contiguous()
                    
                    for avg_pool_matry in self.avg_pool_video:
                        video_enc_temp = avg_pool_matry(video_enc)
                        video_enc_temp = video_enc_temp.transpose(1,2).contiguous()
                        video_enc_list.append(video_enc_temp)
                    return video_enc_list
                else:
                    video_enc = video_enc.transpose(1,2).contiguous()
                    if self.modality == "audiovisual":
                        video_enc = self.avg_pool_video[self.matry_map_video[test_ratio_matry[0]]](video_enc)
                    else:
                        video_enc = self.avg_pool_video[self.matry_map_video[test_ratio_matry]](video_enc)
                    video_enc = video_enc.transpose(1,2).contiguous()
                    return video_enc
        
        else:
        
            if self.downsample_ratio_video != 1:
                if self.compression_mode == "stack":
                    video_enc = [video_enc[:, x:x + self.downsample_ratio_video, :].view(video_enc.shape[0], 1, -1) for x in range(0, video_enc.shape[1], self.downsample_ratio_video)]
                    video_enc = torch.stack(video_enc, dim=1).squeeze(2)
                elif self.compression_mode == "avg-pooling":
                    video_enc = video_enc.transpose(1,2).contiguous()
                    video_enc = self.avg_pool_video(video_enc)
                    video_enc = video_enc.transpose(1,2).contiguous()
                
            return video_enc
    
    def encode_audio(self, audio, max_len, is_trainval, test_ratio_matry = None):
            
        #if is_trainval: # In test time we don't have to convert to float32 and then convert back to bfloat16!!
        audios = audio.to(torch.float32)
        
        audios = audios.cpu().numpy()
        
        
        audio_extract = self.audio_frontend(audios.squeeze(-1), return_tensors="pt",sampling_rate =16000).input_features
        #if is_trainval:
        #    audio_enc = self.audio_encoder(audio_extract.cuda().to(torch.bfloat16)).last_hidden_state
        #else:
        #    audio_enc = self.audio_encoder(audio_extract.cuda()).last_hidden_state
        
        # Next 3 lines for experiments without ASR encoder!!
        #inputs_embeds = nn.functional.gelu(self.audio_encoder.conv1(audio_extract.cuda().to(torch.bfloat16)))
        #audio_enc = nn.functional.gelu(self.audio_encoder.conv2(inputs_embeds))
        #audio_enc = torch.reshape(audio_enc,(-1,audio_enc.shape[-1], audio_enc.shape[1]))
        
        
        audio_enc = self.audio_encoder(audio_extract.cuda().to(torch.bfloat16)).last_hidden_state
    
        # Due to the 30s padding required by Whisper, we drop the tokens that correspond to the padded 0s. As 1s corresponds to 50 tokens, we truncate acccordingly.
        audio_enc = audio_enc[:, 0: max(int(max_len/16000*50), 25) , :]
        
        
        if self.is_matryoshka:
            if self.compression_mode == "stack":
                if is_trainval:
                    audio_enc_list = []
                    for downsample_ratio in self.downsample_ratio_audio:
                        audio_temp = audio_enc
                        audio_enc_ = [audio_temp[:, x:x + downsample_ratio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], downsample_ratio)]
                        rest = audio_temp.shape[1] % downsample_ratio
                        if rest == 0:
                            audio_enc_ = torch.stack(audio_enc_, dim=1).squeeze(2) 
                        else: 
                            audio_enc_ = torch.stack(audio_enc_[:-1], dim=1).squeeze(2)
                        audio_enc_list.append(audio_enc_)
                    return audio_enc_list
                else:
                    if test_ratio_matry != 1:
                        audio_temp = audio_enc
                        if self.modality == "audiovisual":
                            audio_enc = [audio_temp[:, x:x + test_ratio_matry[1], :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], test_ratio_matry[1])]
                            rest = audio_temp.shape[1] % test_ratio_matry[1]
                        else:
                            audio_enc = [audio_temp[:, x:x + test_ratio_matry, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], test_ratio_matry)]
                            rest = audio_temp.shape[1] % test_ratio_matry
                        if rest == 0:
                            audio_enc = torch.stack(audio_enc, dim=1).squeeze(2) 
                        else: 
                            audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
                    return audio_enc
            elif self.compression_mode == "avg-pooling":
                if is_trainval:
                    audio_enc_list = []
                    audio_enc = audio_enc.transpose(1,2).contiguous()
                    
                    for avg_pool_matry in self.avg_pool_audio:
                        audio_enc_temp = avg_pool_matry(audio_enc)
                        audio_enc_temp = audio_enc_temp.transpose(1,2).contiguous()
                        audio_enc_list.append(audio_enc_temp)
                    return audio_enc_list
                else:
                    audio_enc = audio_enc.transpose(1,2).contiguous()
                    if self.modality == "audiovisual":
                        audio_enc = self.avg_pool_audio[self.matry_map_audio[test_ratio_matry[1]]](audio_enc)
                    else:
                        audio_enc = self.avg_pool_audio[self.matry_map_audio[test_ratio_matry]](audio_enc)
                    audio_enc = audio_enc.transpose(1,2).contiguous()
                    return audio_enc   
            
        else:
        
            if self.downsample_ratio_audio != 1:
                if self.compression_mode == "stack":
                    audio_temp = audio_enc
                    audio_enc = [audio_temp[:, x:x + self.downsample_ratio_audio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], self.downsample_ratio_audio)]
                    rest = audio_temp.shape[1] % self.downsample_ratio_audio
                    if rest == 0:
                        audio_enc = torch.stack(audio_enc, dim=1).squeeze(2) 
                    else: 
                        audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
                elif self.compression_mode == "avg-pooling":
                    audio_enc = audio_enc.transpose(1,2).contiguous()
                    audio_enc = self.avg_pool_audio(audio_enc)
                    audio_enc = audio_enc.transpose(1,2).contiguous()
                        
            return audio_enc
        