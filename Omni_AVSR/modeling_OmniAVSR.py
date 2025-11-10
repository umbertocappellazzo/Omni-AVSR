#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:14:59 2025

@author: umbertocappellazzo
"""
import sys
sys.path.append("..")
import torch
from torch import nn
from .Llama_LoRA import LlamaForCausalLM_lora
from .Qwen_LoRA import Qwen2ForCausalLM_lora
from transformers import WhisperModel, LlamaForCausalLM, AutoFeatureExtractor
import fairseq
from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq, AVHubertSeq2SeqConfig
from av_hubert.avhubert.hubert_lora import AVHubertModel_lora
import math
import random

#from av_hubert.avhubert.hubert import AVHubertModel
#from av_hubert.avhubert.hubert_asr import AVHubertSeq2Seq
#from AV_HuBERT_encoder import avhubertConfig, avhubert_encoder

IGNORE_INDEX = -100

class AVSR_LLMs(nn.Module):
    def __init__(self, modality, pretrain_avhubert_enc_video, use_lora_avhubert, llm_model, hidden_size, 
                 intermediate_size, tokenizer, prompt_audio, prompt_video, prompt_audiovisual, pad_id, 
                 downsample_ratio_audio, downsample_ratio_video, audio_encoder_name, compression_mode,
                 unfrozen_modules, max_dec_tokens, num_beams, PETF_LLM_name = None, peft_config_llm = None, 
                 remove_layernorm_from_projector = False, matry_weights = None,
                 is_task_specific = None, is_matryoshka = False, is_single_matry_projector = False
                 ):
        
        super().__init__()
        
        self.modality = modality
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
        self.matry_weights = matry_weights
        self.is_task_specific = is_task_specific
        self.is_matryoshka = is_matryoshka
        self.is_single_matry_projector = is_single_matry_projector
            
        if modality == "audio" or modality == "audiovisual":
                
            print("Instantiating whisper!")    
            self.audio_encoder = WhisperModel.from_pretrained(self.audio_encoder_name).encoder
            self.audio_frontend = AutoFeatureExtractor.from_pretrained(self.audio_encoder_name)
            self.audio_encoder.requires_grad_(False)
            self.audio_encoder.train() # This must be explicitly done as by default the from_pretrained HF models are in eval mode when initialized (this is the opposite for pytorch!)--> cause a break in deepspeed 3! https://github.com/Lightning-AI/pytorch-lightning/issues/19467
            audio_dim =self.audio_encoder.config.hidden_size
            
            if self.compression_mode == "stack":
                
                if self.is_matryoshka:
                    self.matry_map_audio = {}
                    for index, el in enumerate(self.downsample_ratio_audio):
                        self.matry_map_audio[el] = index

                    print("Instantiating stack projector for audio Matryoshka!")
                    
                    if self.remove_layernorm_from_projector:
                        self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size)) for downsample_ratio in self.downsample_ratio_audio])
                    else:
                        self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for downsample_ratio in self.downsample_ratio_audio])
                    
                else:
                
                    print("Instantiating stack projector for audio!")
                    if self.remove_layernorm_from_projector:
                        self.audio_proj = nn.Sequential(nn.Linear(audio_dim*self.downsample_ratio_audio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                    else:
                        self.audio_proj = nn.Sequential(nn.Linear(audio_dim*self.downsample_ratio_audio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
            elif self.compression_mode == "avg-pooling":
                
                if self.is_matryoshka:
                    self.matry_map_audio = {}
                    for index, el in enumerate(self.downsample_ratio_audio):
                        self.matry_map_audio[el] = index
                    
                    print("Instantiating avg-pooling projector for audio Matryoshka!")
                    self.avg_pool_audio = nn.ModuleList([nn.AvgPool1d(downsample_ratio) for downsample_ratio in self.downsample_ratio_audio])
                    if self.remove_layernorm_from_projector:
                        if self.is_single_matry_projector:
                            self.audio_proj = nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                        else:
                            self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for _ in self.downsample_ratio_audio])
                    else:
                        if self.is_single_matry_projector:
                            self.audio_proj = nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
                        else:
                            self.audio_proj = nn.ModuleList([nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size, nn.LayerNorm(hidden_size))) for _ in self.downsample_ratio_audio])
                else:
                
                    self.avg_pool_audio = nn.AvgPool1d(self.downsample_ratio_audio)
                    if self.remove_layernorm_from_projector:
                        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
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
                print("Instantiating stack projector for video!")
                if self.is_matryoshka:
                    self.matry_map_video = {}
                    for index, el in enumerate(self.downsample_ratio_video):
                        self.matry_map_video[el] = index
                    
                    if self.remove_layernorm_from_projector:
                        self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for downsample_ratio in self.downsample_ratio_video])
                    else:
                        self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim*downsample_ratio, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size, nn.LayerNorm(hidden_size))) for downsample_ratio in self.downsample_ratio_video])
                    
                else:
                
                    if self.remove_layernorm_from_projector:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim*self.downsample_ratio_video, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                    else:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim*self.downsample_ratio_video, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))

            elif self.compression_mode == "avg-pooling":
                if self.is_matryoshka:
                    self.matry_map_video = {}
                    for index, el in enumerate(self.downsample_ratio_video):
                        self.matry_map_video[el] = index
                    
                    print("Instantiating avg-pooling projector for video Matryoshka!")
                    self.avg_pool_video = nn.ModuleList([nn.AvgPool1d(downsample_ratio) for downsample_ratio in self.downsample_ratio_video])
                    if self.remove_layernorm_from_projector:
                        if self.is_single_matry_projector:
                            self.video_proj = nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                        else:
                            self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size)) for _ in self.downsample_ratio_video])
                    else:
                        if self.is_single_matry_projector:
                            self.video_proj = nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
                        else:
                            self.video_proj = nn.ModuleList([nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size, nn.LayerNorm(hidden_size))) for _ in self.downsample_ratio_video])
                
                else:
                    print("Instantiating avg-pooling projector for video!")
                    self.avg_pool_video = nn.AvgPool1d(self.downsample_ratio_video)
                    if self.remove_layernorm_from_projector:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size))
                    else:
                        self.video_proj = nn.Sequential(nn.Linear(video_dim, intermediate_size), nn.ReLU(), nn.Linear(intermediate_size, hidden_size), nn.LayerNorm(hidden_size))
        
        if "llama" in llm_model:
            if self.PETF_LLM_name is None:
                self.llm = LlamaForCausalLM.from_pretrained(llm_model)
            else:
                assert self.PETF_LLM_name == "lora"
                self.llm = LlamaForCausalLM_lora.from_pretrained(llm_model, peft_config_llm) 
               
        elif "Qwen" in llm_model:
            if self.PETF_LLM_name == "lora":
                self.llm = Qwen2ForCausalLM_lora.from_pretrained(llm_model, peft_config_llm) 
        
        # IMPORTANT: we need to add the pad_id to the model and resize the token embeddings matrix accordingly.
        self.tokenizer = tokenizer
        if "llama" in llm_model:
            self.llm.config.pad_token_id = pad_id
        
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        self.llm.requires_grad_(False)
        
        prompt_tokens_start_at = 0 if "Qwen" in self.llm_model else 1
        self.register_buffer("prompt_audio", self.llm.model.embed_tokens(self.tokenizer(prompt_audio, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1]))
        self.register_buffer("prompt_video", self.llm.model.embed_tokens(self.tokenizer(prompt_video, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1]))
        self.register_buffer("prompt_audiovisual", self.llm.model.embed_tokens(self.tokenizer(prompt_audiovisual, return_tensors = "pt").input_ids[:,prompt_tokens_start_at:-1]))
        
        self.prompt_audio_len = self.prompt_audio.shape[1]
        self.prompt_video_len = self.prompt_video.shape[1]
        self.prompt_audiovisual_len = self.prompt_audiovisual.shape[1]
        
        print(f"The audio prompt has {self.prompt_audio_len} tokens.")
        print(f"The video prompt has {self.prompt_video_len} tokens.")
        print(f"The audiovisual prompt has {self.prompt_audiovisual_len} tokens.")
        
        self._unfreeze_PETF(unfrozen_modules)
        
        
    def _unfreeze_PETF(self, unfrozen_modules):
        """
        Modules to be unfrozen. Unfrozen_blocks is a list with one or multiple values: ['peft_audio','peft_video','embedding','peft_llm']. 
        """
        if None in unfrozen_modules:
            return
        if "peft_llm" in unfrozen_modules:
            print("Unfreezing LoRA for LLM:")
            for block_idx in range(self.llm.config.num_hidden_layers):
                self.llm.model.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                self.llm.model.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
                
                if self.peft_config_llm.SHARED_LORA:
                    self.llm.model.layers[block_idx].self_attn.lora_down_Q_shared.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_up_Q_shared.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_down_V_shared.requires_grad_(True)
                    self.llm.model.layers[block_idx].self_attn.lora_up_V_shared.requires_grad_(True)
        
        if "lora_avhubert" in unfrozen_modules:
            print("Unfreezing LoRA for AV-HuBERT video encoder!")
            for block_idx in range(24):
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_Q.requires_grad_(True)
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_Q.requires_grad_(True)
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_down_V.requires_grad_(True)
                self.video_encoder.encoder.layers[block_idx].self_attn.lora_up_V.requires_grad_(True)
            
        
    def forward(self, inputs, is_trainval= True, modality = None, test_ratio_matry_audio = None, test_ratio_matry_video = None):
        
        if is_trainval:
            output_dict = self.prepare_inputs(inputs, is_trainval, test_ratio_matry_audio = test_ratio_matry_audio, test_ratio_matry_video = test_ratio_matry_video)
            
            if self.is_task_specific:
                if "Qwen" in self.llm_model:
                    audio_output = self.llm(inputs_embeds = torch.cat([output_dict["audio_tokens"], self.prompt_audio.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"]], dim = 1),
                                            labels = output_dict["labels_audio"], modality = "audio")
                    video_output = self.llm(inputs_embeds = torch.cat([output_dict["video_tokens"], self.prompt_video.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"]], dim = 1),
                                            labels = output_dict["labels_video"], modality = "video")
                    audiovisual_output = self.llm(inputs_embeds = torch.cat([output_dict["audio_tokens"], output_dict["video_tokens"], self.prompt_audiovisual.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"]], dim = 1),
                                            labels = output_dict["labels_audiovisual"], modality = "audiovisual")
                    
                else:
                    audio_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], self.prompt_audio.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                            labels = output_dict["labels_audio"], modality = "audio")
                    video_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["video_tokens"], self.prompt_video.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                            labels = output_dict["labels_video"], modality = "video")
                    audiovisual_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], output_dict["video_tokens"], self.prompt_audiovisual.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                                  labels = output_dict["labels_audiovisual"], modality = "audiovisual")
            else:
                if "Qwen" in self.llm_model:
                    audio_output = self.llm(inputs_embeds = torch.cat([output_dict["audio_tokens"], self.prompt_audio.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"]], dim = 1),
                                            labels = output_dict["labels_audio"])
                    video_output = self.llm(inputs_embeds = torch.cat([output_dict["video_tokens"], self.prompt_video.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"]], dim = 1),
                                            labels = output_dict["labels_video"])
                    audiovisual_output = self.llm(inputs_embeds = torch.cat([output_dict["audio_tokens"], output_dict["video_tokens"], self.prompt_audiovisual.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"]], dim = 1),
                                            labels = output_dict["labels_audiovisual"])
                    
                else:
                    audio_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], self.prompt_audio.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                            labels = output_dict["labels_audio"])
                    video_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["video_tokens"], self.prompt_video.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                            labels = output_dict["labels_video"])
                    audiovisual_output = self.llm(inputs_embeds = torch.cat([output_dict["text_embeddings"][:,0,:].unsqueeze(1), output_dict["audio_tokens"], output_dict["video_tokens"], self.prompt_audiovisual.expand(inputs["tokens"].shape[0],-1,-1), output_dict["text_embeddings"][:, 1:, :]], dim = 1),
                                                  labels = output_dict["labels_audiovisual"])
                
            
            audio_loss = audio_output.loss*self.matry_weights[0] if self.matry_weights else audio_output.loss
            video_loss = video_output.loss*self.matry_weights[1] if self.matry_weights else video_output.loss
            audiovisual_loss = audiovisual_output.loss*self.matry_weights[2] if self.matry_weights else audiovisual_output.loss
                
            return audio_loss, video_loss, audiovisual_loss
        
        else:
               
            embeddings = self.prepare_inputs(inputs, is_trainval, test_ratio_matry_audio = test_ratio_matry_audio, test_ratio_matry_video = test_ratio_matry_video) 
           
            if self.llm_model ==  "meta-llama/Meta-Llama-3-8B" or self.llm_model == "meta-llama/Meta-Llama-3.1-8B" or self.llm_model == "meta-llama/Llama-3.2-1B" or self.llm_model == "meta-llama/Llama-3.2-3B":
                decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<|end_of_text|>"], 
                                                    bos_token_id = self.tokenizer.vocab["<|begin_of_text|>"], 
                                                    pad_token_id = self.tokenizer.vocab["<pad>"],
                                                    modality = modality
                                                    )
            elif "Qwen" in self.llm_model:
                decoded_ids = self.llm.generate(inputs_embeds = embeddings, max_new_tokens = self.max_dec_tokens, num_beams=self.num_beams, eos_token_id = self.tokenizer.vocab["<|endoftext|>"], 
                                                pad_token_id = self.tokenizer.vocab["<|endoftext|>"],
                                                modality = modality
                                                )
            return decoded_ids
            
    
    def prepare_inputs(self, inputs, is_trainval, test_ratio_matry_audio = None, test_ratio_matry_video = None):
        
        
        if is_trainval:
            if self.is_matryoshka:
                audio_features, selected_audio_rate = self.encode_audio(inputs["audio"], max(inputs["lengths"]), is_trainval = is_trainval, test_ratio_matry_audio = test_ratio_matry_audio)
                video_features, selected_video_rate = self.encode_video(inputs["video"], is_trainval = is_trainval, test_ratio_matry_video = test_ratio_matry_video)
            else: 
                audio_features = self.encode_audio(inputs["audio"], max(inputs["lengths"]))
                video_features = self.encode_video(inputs["video"])
            
            text_embeddings = self.llm.model.embed_tokens(inputs["tokens"])
            
            ignore_count_audio = 0 
            ignore_count_video = 0
            ignore_count_audiovisual = 0
            
            ignore_count_audio += self.prompt_audio_len
            ignore_count_video += self.prompt_video_len
            ignore_count_audiovisual += self.prompt_audiovisual_len
            
            video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            video_starts =  self.llm.model.embed_tokens(video_starts)
            
            video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            video_ends = self.llm.model.embed_tokens(video_ends)
            
            video_features = self.video_proj(video_features) if self.is_single_matry_projector else self.video_proj[self.matry_map_video[selected_video_rate]](video_features)
            
            video_inputs = torch.cat([video_starts, video_features, video_ends], dim=1)
            
            ignore_count_video += video_inputs.shape[1]
            ignore_count_audiovisual += video_inputs.shape[1]
            
            audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            audio_starts =  self.llm.model.embed_tokens(audio_starts)
            
            audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
            audio_ends = self.llm.model.embed_tokens(audio_ends)
            
            audio_features = self.audio_proj(audio_features) if self.is_single_matry_projector else self.audio_proj[self.matry_map_audio[selected_audio_rate]](audio_features)
        
            audio_inputs = torch.cat([audio_starts, audio_features, audio_ends], dim=1)
            
            ignore_count_audio += audio_inputs.shape[1]
            ignore_count_audiovisual += audio_inputs.shape[1]
            
            labels_audio = torch.tensor([IGNORE_INDEX]*ignore_count_audio, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
            labels_video = torch.tensor([IGNORE_INDEX]*ignore_count_video, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
            labels_audiovisual = torch.tensor([IGNORE_INDEX]*ignore_count_audiovisual, device=text_embeddings.device).expand(text_embeddings.shape[0], -1)
            
            if "Qwen" in self.llm_model:
                labels_audio = torch.cat([labels_audio, inputs["labels"]], dim=1)
                labels_video = torch.cat([labels_video, inputs["labels"]], dim=1)
                labels_audiovisual = torch.cat([labels_audiovisual, inputs["labels"]], dim=1)
            else:
                labels_audio = torch.cat(
                    [inputs["labels"][:, 0].unsqueeze(1), labels_audio, inputs["labels"][:, 1:]], dim=1)
                labels_video = torch.cat(
                    [inputs["labels"][:, 0].unsqueeze(1), labels_video, inputs["labels"][:, 1:]], dim=1)
                labels_audiovisual = torch.cat(
                    [inputs["labels"][:, 0].unsqueeze(1), labels_audiovisual, inputs["labels"][:, 1:]], dim=1)
            
            return {"text_embeddings": text_embeddings,
                    "audio_tokens": audio_inputs,
                    "video_tokens": video_inputs,
                    "labels_audio": labels_audio,
                    "labels_video": labels_video,
                    "labels_audiovisual": labels_audiovisual
                    }
            
        else:
            if self.is_matryoshka:
                audio_features = self.encode_audio(inputs["audio"], max(inputs["lengths"]), is_trainval = is_trainval, test_ratio_matry_audio = test_ratio_matry_audio) if self.modality in ["audio", "audiovisual"] else None
                video_features = self.encode_video(inputs["video"], is_trainval = is_trainval, test_ratio_matry_video = test_ratio_matry_video) if self.modality in ["video", "audiovisual"] else None
            else:
                audio_features = self.encode_audio(inputs["audio"], max(inputs["lengths"])) if self.modality in ["audio", "audiovisual"] else None
                video_features = self.encode_video(inputs["video"]) if self.modality in ["video", "audiovisual"] else None
            
            
            text_embeddings_ = self.llm.model.embed_tokens(inputs["tokens"])
            
            #prompt_tokens_start_at = 0 if "Qwen" in self.llm_model else 1
            if self.modality == "audio":
                prompt_ids = self.prompt_audio
            elif self.modality == "video":
                prompt_ids = self.prompt_video
            else:
                prompt_ids = self.prompt_audiovisual#.device)
        
            if "Qwen" in self.llm_model:
                    text_embeddings = prompt_ids
            else:
                text_embeddings = torch.cat([text_embeddings_[:, 0, :].unsqueeze(1), prompt_ids], dim=1)
        
            if video_features is not None:
                video_starts = torch.tensor([self.tokenizer.vocab["<video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_starts =  self.llm.model.embed_tokens(video_starts)
                
                video_ends = torch.tensor([self.tokenizer.vocab["</video>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                video_ends = self.llm.model.embed_tokens(video_ends)
                
                
                video_features = self.video_proj(video_features) if self.is_single_matry_projector else self.video_proj[self.matry_map_video[test_ratio_matry_video]](video_features)
                
                video_inputs = torch.cat([video_starts, video_features, video_ends], dim=1)
               
                if "Qwen" in self.llm_model:
                    text_embeddings = torch.cat([video_inputs, text_embeddings], dim=1)
                else:
                    text_embeddings = torch.cat(
                        [text_embeddings[:, 0, :].unsqueeze(1), video_inputs, text_embeddings[:, 1:, :]], dim=1)
                    
            
            if audio_features is not None:
                audio_starts = torch.tensor([self.tokenizer.vocab["<audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_starts =  self.llm.model.embed_tokens(audio_starts)
                
                audio_ends = torch.tensor([self.tokenizer.vocab["</audio>"]], device = text_embeddings.device).expand(inputs["tokens"].shape[0],-1)
                audio_ends = self.llm.model.embed_tokens(audio_ends)
                
                audio_features = self.audio_proj(audio_features) if self.is_single_matry_projector else self.audio_proj[self.matry_map_audio[test_ratio_matry_audio]](audio_features)
            
                audio_inputs = torch.cat([audio_starts, audio_features, audio_ends], dim=1)
                
                if "Qwen" in self.llm_model:
                    text_embeddings = torch.cat([audio_inputs, text_embeddings], dim=1)
                else:
                    text_embeddings = torch.cat(
                        [text_embeddings[:, 0, :].unsqueeze(1), audio_inputs,text_embeddings[:, 1:, :] ], dim=1)
            
            
            return text_embeddings
        
    
    def encode_video(self, videos, is_trainval = None, test_ratio_matry_video = None):
        
        video_enc, _, encoder_layers = self.video_encoder.extract_finetune(source={'video': torch.reshape(videos,(-1,videos.shape[2],videos.shape[1],videos.shape[3],videos.shape[-1])),'audio': None})
        
        if self.is_matryoshka:
            if self.compression_mode == "avg-pooling":
                if is_trainval:
                    if test_ratio_matry_video:
                        video_enc = video_enc.transpose(1,2).contiguous()
                        video_enc = self.avg_pool_video[self.matry_map_video[test_ratio_matry_video]](video_enc)
                        video_enc = video_enc.transpose(1,2).contiguous()
                        return video_enc, test_ratio_matry_video
                    else:
                        current_ratio = random.choice(self.downsample_ratio_video) # We choose a random rate.
                        video_enc = video_enc.transpose(1,2).contiguous()
                        video_enc = self.avg_pool_video[self.matry_map_video[current_ratio]](video_enc)
                        video_enc = video_enc.transpose(1,2).contiguous()
                        return video_enc, current_ratio
                else:
                     video_enc = video_enc.transpose(1,2).contiguous()
                     video_enc = self.avg_pool_video[self.matry_map_video[test_ratio_matry_video]](video_enc)
                     video_enc = video_enc.transpose(1,2).contiguous()
                     return video_enc
            else:
                if is_trainval:
                    if test_ratio_matry_video:
                        video_temp = video_enc
                        video_enc = [video_temp[:, x:x + test_ratio_matry_video, :].view(video_temp.shape[0], 1, -1) for x in range(0, video_temp.shape[1], test_ratio_matry_video)]
                        rest = video_temp.shape[1] % test_ratio_matry_video
                        if rest == 0:
                            video_enc = torch.stack(video_enc, dim=1).squeeze(2) 
                        else: 
                            video_enc = torch.stack(video_enc[:-1], dim=1).squeeze(2)
                        return video_enc, test_ratio_matry_video
                    else:
                        video_temp = video_enc
                        current_ratio = random.choice(self.downsample_ratio_video) # We choose a random rate.
                        video_enc = [video_temp[:, x:x + current_ratio, :].view(video_temp.shape[0], 1, -1) for x in range(0, video_temp.shape[1], current_ratio)]
                        
                        rest = video_temp.shape[1] % current_ratio
                        if rest == 0:
                            video_enc = torch.stack(video_enc, dim=1).squeeze(2) 
                        else: 
                            video_enc = torch.stack(video_enc[:-1], dim=1).squeeze(2)
                        return video_enc, current_ratio
                else:
                    video_temp = video_enc
                    video_enc = [video_temp[:, x:x + test_ratio_matry_video, :].view(video_temp.shape[0], 1, -1) for x in range(0, video_temp.shape[1], test_ratio_matry_video)]
                    rest = video_temp.shape[1] % test_ratio_matry_video
                    if rest == 0:
                        video_enc = torch.stack(video_enc, dim=1).squeeze(2) 
                    else: 
                        video_enc = torch.stack(video_enc[:-1], dim=1).squeeze(2)
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
    
    def encode_audio(self, audio, max_len, is_trainval = None, test_ratio_matry_audio = None):
            
        #if is_trainval: # In test time we don't have to convert to float32 and then convert back to bfloat16!
        audios = audio.to(torch.float32)
        audios = audios.cpu().numpy()
        audio_extract = self.audio_frontend(audios.squeeze(-1), return_tensors="pt",sampling_rate =16000).input_features
        audio_enc = self.audio_encoder(audio_extract.cuda().to(torch.bfloat16)).last_hidden_state
    
        # Due to the 30s padding required by Whisper, we drop the tokens that correspond to the padded 0s. As 1s corresponds to 50 tokens, we truncate acccordingly.
        audio_enc = audio_enc[:, 0: max(int(max_len/16000*50), 25) , :]
        
        
        if self.is_matryoshka:
            if self.compression_mode == "avg-pooling":
                if is_trainval:
                    if test_ratio_matry_audio:
                        audio_enc = audio_enc.transpose(1,2).contiguous()
                        audio_enc = self.avg_pool_audio[self.matry_map_audio[test_ratio_matry_audio]](audio_enc) 
                        audio_enc = audio_enc.transpose(1,2).contiguous()
                        return audio_enc, test_ratio_matry_audio
                    else:
                        current_ratio = random.choice(self.downsample_ratio_audio) # We choose a random rate.
                        audio_enc = audio_enc.transpose(1,2).contiguous()
                        audio_enc = self.avg_pool_audio[self.matry_map_audio[current_ratio]](audio_enc)
                        audio_enc = audio_enc.transpose(1,2).contiguous()
                        return audio_enc, current_ratio
                else:
                    audio_enc = audio_enc.transpose(1,2).contiguous()
                    audio_enc = self.avg_pool_audio[self.matry_map_audio[test_ratio_matry_audio]](audio_enc) 
                    audio_enc = audio_enc.transpose(1,2).contiguous()
                    return audio_enc
            else:
                if is_trainval:
                    if test_ratio_matry_audio:
                        audio_temp = audio_enc
                        audio_enc = [audio_temp[:, x:x + test_ratio_matry_audio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], test_ratio_matry_audio)]
                        rest = audio_temp.shape[1] % test_ratio_matry_audio
                        if rest == 0:
                            audio_enc = torch.stack(audio_enc, dim=1).squeeze(2) 
                        else: 
                            audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
                        return audio_enc, test_ratio_matry_audio
                    else:
                        current_ratio = random.choice(self.downsample_ratio_audio) # We choose a random rate.
                        audio_temp = audio_enc
                        audio_enc = [audio_temp[:, x:x + current_ratio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], current_ratio)]
                        rest = audio_temp.shape[1] % current_ratio
                        if rest == 0:
                            audio_enc = torch.stack(audio_enc, dim=1).squeeze(2) 
                        else: 
                            audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
                        return audio_enc, current_ratio
                else:
                    audio_temp = audio_enc
                    audio_enc = [audio_temp[:, x:x + test_ratio_matry_audio, :].view(audio_temp.shape[0], 1, -1) for x in range(0, audio_temp.shape[1], test_ratio_matry_audio)]
                    rest = audio_temp.shape[1] % test_ratio_matry_audio
                    if rest == 0:
                        audio_enc = torch.stack(audio_enc, dim=1).squeeze(2) 
                    else: 
                        audio_enc = torch.stack(audio_enc[:-1], dim=1).squeeze(2)
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