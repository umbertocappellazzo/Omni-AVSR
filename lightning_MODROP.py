#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 13:37:41 2025

@author: umbertocappellazzo
"""

import torch
import torchaudio
from cosine import WarmupCosineScheduler
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from Llama_adapters import Adapter_config_llama
#from Llama_MatryAdapter import MatryAdapter_config
#from Llama_MatryMoELoRA import MatryMoELoRA_config
#from Llama_MatryMoEAdapter import MatryMoEAdapter_config
from Llama_MoEAdapter import MoEAdapter_config
from Llama_LoRA import LoRA_config
#from Llama_MatryLoRA import MatryLoRA_config
from Qwen_LoRA import QwenLoRA_config
#from Llama_Omni_SMoLA import Omni_SMoLA_config
from modeling_MODROP import AVSR_LLMs
#from modeling_Mixture_of_Whsipers import MoW_LLM
from tokenizers.processors import TemplateProcessing

#import torch.nn.functional as F


#import time
#import datetime

DEFAULT_PAD_TOKEN = "<pad>"
AUDIO_SOS = "<audio>"
AUDIO_EOS = "</audio>"
VIDEO_SOS = "<video>"
VIDEO_EOS = "</video>"
#PROMPT = "Transcribe video to text."



def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)

class ModelModule_LLM(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        if args.use_lora_avhubert:
            assert "lora_avhubert" in args.unfrozen_modules, ("LoRA modules for the AV-HuBERT encoder must be unfrozen!!")
        
        add_bos_token = "Qwen" not in args.llm_model # Qwen doesn't have bos token
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model, add_bos_token=add_bos_token, add_eos_token= True)
        
        # Apparently, some LLMs don't rely on FastTokenizer and it seems like they don't append the EOS token even though you set
        # it explicitly. In my case, this happens for LLama3. More details at: https://github.com/huggingface/transformers/issues/22794.
        
        if args.llm_model == "meta-llama/Meta-Llama-3-8B" or args.llm_model == "meta-llama/Meta-Llama-3.1-8B" or args.llm_model == "meta-llama/Llama-3.2-1B" or args.llm_model == "meta-llama/Llama-3.2-3B":
            bos = self.tokenizer.bos_token
            eos = self.tokenizer.eos_token
            
            self.tokenizer._tokenizer.post_processor =TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[
                    (f"{bos}", self.tokenizer.bos_token_id), 
                    (f"{eos}", self.tokenizer.eos_token_id)
                    ],
                )
        elif args.llm_model in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]:
            eos = self.tokenizer.eos_token

            self.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"$A:0 {eos}:0",
                pair=f"$A:0 {eos}:0 $B:1 {eos}:1",
                special_tokens=[
                    (f"{eos}", self.tokenizer.eos_token_id)
                    ],
                )
        
        # By default, LLaMA doesn't come with a padding token (pad_token= None), so we need to introduce it.
        if 'llama' in args.llm_model or "Tiny" in args.llm_model or "mistral" in args.llm_model:
            num_added_toks = self.tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN, "additional_special_tokens": [AUDIO_SOS, AUDIO_EOS, VIDEO_SOS, VIDEO_EOS]})
            pad_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN)
        else:
            num_added_toks = self.tokenizer.add_special_tokens({"additional_special_tokens": [AUDIO_SOS, AUDIO_EOS, VIDEO_SOS, VIDEO_EOS]})
            pad_id = None
            
        print("We have added ", num_added_toks, " tokens to the tokenizer!")
        self.tokenizer.padding_side = "right"   # The padding is added to the right.
        
        # The resize of the embed_tokens matrix and the add of the pad_token to the model is performed when the model is called.
        
        
        prompt_audio = args.prompt_audio
        prompt_video = args.prompt_video
        prompt_audiovisual = args.prompt_audiovisual
        
        #if args.modality == 'audio':
        #    prompt = args.prompt_audio
        #elif args.modality == 'video':
        #    prompt = args.prompt_video
        #else:
        #    assert args.modality in ['audiovisual', 'audiovisual_avhubert']
        #    prompt = args.prompt_audiovisual
        
        
        if args.add_PETF_LLM:
            
            if args.add_PETF_LLM == 'adapter':
                
                adapter_config_llm = MoEAdapter_config(args.reduction_rate_adapter, args.n_experts, args.topk, args.adapter_location, args.num_shared_experts, args.apply_load_balancing_loss, args.is_hydra) if args.is_MoE else Adapter_config_llama(args.reduction_rate_adapter, args.adapter_location) 
                
                self.model = AVSR_LLMs(modality = args.modality, 
                                       is_MoE = args.is_MoE,   
                                       pretrain_avhubert_enc_video = args.pretrain_avhubert_enc_video_path, 
                                       use_lora_avhubert= args.use_lora_avhubert,
                                       llm_model = args.llm_model, 
                                       hidden_size = args.hidden_size, 
                                       intermediate_size= args.intermediate_size, 
                                       tokenizer = self.tokenizer, 
                                       prompt_audio = prompt_audio, 
                                       prompt_video = prompt_video, 
                                       prompt_audiovisual = prompt_audiovisual, 
                                       pad_id = pad_id, 
                                       downsample_ratio_audio = args.downsample_ratio_audio, 
                                       downsample_ratio_video = args.downsample_ratio_video, 
                                       audio_encoder_name = args.audio_encoder_name, 
                                       unfrozen_modules= args.unfrozen_modules, 
                                       max_dec_tokens = args.max_dec_tokens, 
                                       num_beams = args.num_beams, 
                                       PETF_LLM_name = args.add_PETF_LLM, 
                                       peft_config_llm= adapter_config_llm, 
                                       compression_mode= args.compression_mode,
                                       remove_layernorm_from_projector = args.no_layernorm_projector,
                                       probs= args.probs
                                       )
            
            
            elif args.add_PETF_LLM == 'lora':
                if "Qwen" in args.llm_model:
                    IS_QWEN25_0_5B = True if args.llm_model == "Qwen/Qwen2.5-0.5B" else False
                    IS_QWEN25_1_5B = True if args.llm_model == "Qwen/Qwen2.5-1.5B" else False
                    IS_QWEN25_3B = True if args.llm_model == "Qwen/Qwen2.5-3B" else False
                    IS_QWEN25_7B = True if args.llm_model == "Qwen/Qwen2.5-7B" else False
                    lora_config_llm = QwenLoRA_config(args.rank, args.alpha, IS_QWEN25_0_5B, IS_QWEN25_1_5B, IS_QWEN25_3B, IS_QWEN25_7B)
                else:
                    IS_LLAMA3 = True if args.llm_model == "meta-llama/Meta-Llama-3-8B" or args.llm_model == "meta-llama/Meta-Llama-3.1-8B" or args.llm_model == "mistralai/Mistral-7B-v0.1" or args.llm_model == "meta-llama/Llama-3.2-1B" else False
                    IS_LLAMA3_2_3B = True if args.llm_model == "meta-llama/Llama-3.2-3B" else False
                    IS_TINYLLAMA = True if args.llm_model == "TinyLlama/TinyLlama_v1.1" else False
                    
                    lora_config_llm = LoRA_config(args.rank, args.alpha, IS_LLAMA3, IS_TINYLLAMA, IS_LLAMA3_2_3B, args.is_task_specific)
                    
                    
                self.model = AVSR_LLMs(modality = args.modality, 
                                       is_MoE = args.is_MoE,   
                                       pretrain_avhubert_enc_video = args.pretrain_avhubert_enc_video_path, 
                                       use_lora_avhubert= args.use_lora_avhubert,
                                       llm_model = args.llm_model, 
                                       hidden_size = args.hidden_size, 
                                       intermediate_size= args.intermediate_size, 
                                       tokenizer = self.tokenizer, 
                                       prompt_audio = prompt_audio, 
                                       prompt_video = prompt_video, 
                                       prompt_audiovisual = prompt_audiovisual, 
                                       pad_id = pad_id, 
                                       downsample_ratio_audio = args.downsample_ratio_audio, 
                                       downsample_ratio_video = args.downsample_ratio_video, 
                                       audio_encoder_name = args.audio_encoder_name, 
                                       unfrozen_modules= args.unfrozen_modules, 
                                       max_dec_tokens = args.max_dec_tokens, 
                                       num_beams = args.num_beams, 
                                       PETF_LLM_name = args.add_PETF_LLM, 
                                       peft_config_llm= lora_config_llm,
                                       compression_mode= args.compression_mode,
                                       remove_layernorm_from_projector = args.no_layernorm_projector,
                                       probs= args.probs,
                                       is_task_specific = args.is_task_specific
                                       )
            
            self.model._unfreeze_PETF(args.unfrozen_modules)
            n_parameters_learn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters of the model: ", n_parameters_learn)
                
        
        # initialize the full model from the checkpoint for inference.
        if args.pretrained_model_path:
            ckpt = torch.load(args.pretrained_model_path)#["state_dict"]
            #ckpt = {k[6:]: v for k, v in ckpt.items() if k.startswith("model.")}
            self.model.load_state_dict(ckpt) # strict = False
            #print("LoRA Q down layer 0 params after: ",self.model.llm.model.layers[0].self_attn.lora_down_Q.weight[0,0:10])
            
        
    def configure_optimizers(self):
        #optimizer = DeepSpeedCPUAdam(model_params= self.parameters(), lr= self.args.lr, betas=(0.9, 0.98), weight_decay=self.args.weight_decay)
        #optimizer = FusedAdam(params= self.parameters(), lr= self.args.lr, betas=(0.9, 0.98), weight_decay=self.args.weight_decay)
        #optimizer = torch.optim.AdamW([{'params':self.model.video_encoder.parameters(),'lr':5e-4}],lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        
        #optimizer = torch.optim.AdamW([{'params':self.model.video_proj.parameters(),'lr':5e-3}],lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        optimizer = torch.optim.AdamW(params= self.model.parameters(), lr= self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (len(self.trainer.datamodule.train_dataloader())))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        
        if self.args.is_MoE and self.args.apply_load_balancing_loss:
            moe_loss, load_balancing_loss = self.model(batch, mode = "train")
            train_loss = moe_loss + self.args.load_balancing_loss_coeff*load_balancing_loss
        else:
            train_loss = self.model(batch, mode = "train")
        
        batch_size = batch["tokens"].shape[0]
        #leng = sum(batch["lengths"])
        
        #self.log("lengths", leng, on_step=True)
        self.log("loss", train_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        if self.args.is_MoE and self.args.apply_load_balancing_loss:
            #self.log("llm_loss", moe_loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("load_balancing_loss", load_balancing_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        
        batch_sizes = self.all_gather(batch_size)
        #self.log("batch size", batch_size, on_step=True)
        
        
        
        train_loss *= batch_sizes.size(0) / batch_sizes.sum()
        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))
        
        return train_loss
            
        
    def validation_step(self, batch, batch_idx):
        if self.args.is_MoE and self.args.apply_load_balancing_loss:
            val_moe_loss_audio, val_moe_loss_video, val_moe_loss_audiovisual, val_load_balancing_loss_audio, val_load_balancing_loss_video, val_load_balancing_loss_audiovisual = self.model(batch, mode = "val")
            val_loss_audio = val_moe_loss_audio + self.args.load_balancing_loss_coeff*val_load_balancing_loss_audio
            val_loss_video = val_moe_loss_video + self.args.load_balancing_loss_coeff*val_load_balancing_loss_video
            val_loss_audiovisual = val_moe_loss_audiovisual + self.args.load_balancing_loss_coeff*val_load_balancing_loss_audiovisual
            val_loss = (val_loss_audio + val_loss_video + val_loss_audiovisual)/3
        else:
            val_loss_audio, val_loss_video, val_loss_audiovisual = self.model(batch, mode = "val")
            val_loss = (val_loss_audio + val_loss_video + val_loss_audiovisual)/3
        
        batch_size = batch["tokens"].shape[0]
        
        self.log("loss_val", val_loss, batch_size=batch_size, sync_dist=True)
        self.log("loss_val_audio", val_loss_audio, batch_size=batch_size, sync_dist=True)
        self.log("loss_val_video", val_loss_video, batch_size=batch_size, sync_dist=True)
        self.log("loss_val_audiovisual", val_loss_audiovisual, batch_size=batch_size, sync_dist=True)
        if self.args.is_MoE and self.args.apply_load_balancing_loss:
            #self.log("val_llm_loss", val_moe_loss, batch_size=batch_size, sync_dist=True)
            self.log("val_load_balancing_loss_audio", val_load_balancing_loss_audio, batch_size=batch_size, sync_dist=True)
            self.log("val_load_balancing_loss_video", val_load_balancing_loss_video, batch_size=batch_size, sync_dist=True)
            self.log("val_load_balancing_loss_audiovisual", val_load_balancing_loss_audiovisual, batch_size=batch_size, sync_dist=True)
        return val_loss
    
    
    
    
    def test_step(self, batch, batch_idx):
        
        generated_ids = self.model(batch, mode = "test", modality = self.model.modality) if self.args.is_task_specific else self.model(batch, mode = "test")

        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        self.total_edit_distance += compute_word_level_distance(batch["gold_text"], generated_text)
        self.total_length += len(batch["gold_text"].split())
        
        
        return

    
    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        
        print(f"Setting {self.args.modality} modality to the model.")
        self.model.modality = self.args.modality
        
    def on_test_epoch_end(self):
        
        self.log("wer", self.total_edit_distance / self.total_length)