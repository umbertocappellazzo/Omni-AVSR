#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:16:24 2025

@author: umbertocappellazzo
"""

import torch
import torchaudio
from cosine import WarmupCosineScheduler
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from Llama_LoRA import LoRA_config
from Qwen_LoRA import QwenLoRA_config
from modeling_OmniAVSR import AVSR_LLMs
from tokenizers.processors import TemplateProcessing
import torch.nn.functional as F

DEFAULT_PAD_TOKEN = "<pad>"
AUDIO_SOS = "<audio>"
AUDIO_EOS = "</audio>"
VIDEO_SOS = "<video>"
VIDEO_EOS = "</video>"

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
        elif "Qwen" in args.llm_model:
            eos = self.tokenizer.eos_token

            self.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"$A:0 {eos}:0",
                pair=f"$A:0 {eos}:0 $B:1 {eos}:1",
                special_tokens=[
                    (f"{eos}", self.tokenizer.eos_token_id)
                    ],
                )
        
        # By default, LLaMA doesn't come with a padding token (pad_token= None), so we need to introduce it.
        if 'llama' in args.llm_model:
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
        
        if args.add_PETF_LLM:
            
            if "Qwen" in args.llm_model:
                IS_QWEN25_0_5B = True if args.llm_model == "Qwen/Qwen2.5-0.5B" else False
                IS_QWEN25_1_5B = True if args.llm_model == "Qwen/Qwen2.5-1.5B" else False
                IS_QWEN25_3B = True if args.llm_model == "Qwen/Qwen2.5-3B" else False
                IS_QWEN25_7B = True if args.llm_model == "Qwen/Qwen2.5-7B" else False
                IS_QWEN25_14B = True if args.llm_model == "Qwen/Qwen2.5-14B" else False
                IS_QWEN25_32B = True if args.llm_model == "Qwen/Qwen2.5-32B" else False
                lora_config_llm = QwenLoRA_config(args.rank, args.alpha, IS_QWEN25_0_5B, IS_QWEN25_1_5B, IS_QWEN25_3B, IS_QWEN25_7B, IS_QWEN25_14B, IS_QWEN25_32B, args.is_task_specific, args.use_shared_lora_task_specific)
            else:
                IS_LLAMA3 = True if args.llm_model == "meta-llama/Meta-Llama-3-8B" or args.llm_model == "meta-llama/Meta-Llama-3.1-8B" or args.llm_model == "meta-llama/Llama-3.2-1B" else False
                IS_LLAMA3_2_3B = True if args.llm_model == "meta-llama/Llama-3.2-3B" else False
                
                lora_config_llm = LoRA_config(args.rank, args.alpha, IS_LLAMA3, IS_LLAMA3_2_3B, args.is_task_specific, args.use_shared_lora_task_specific)
                
            self.model = AVSR_LLMs(modality = args.modality, 
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
                                   compression_mode= args.compression_mode,
                                   unfrozen_modules= args.unfrozen_modules, 
                                   max_dec_tokens = args.max_dec_tokens, 
                                   num_beams = args.num_beams, 
                                   PETF_LLM_name = args.add_PETF_LLM, 
                                   peft_config_llm= lora_config_llm,
                                   remove_layernorm_from_projector = args.no_layernorm_projector,
                                   matry_weights = args.matry_weights,
                                   is_task_specific = args.is_task_specific,
                                   is_matryoshka = args.is_matryoshka,
                                   is_single_matry_projector = args.is_single_matry_projector,
                                   )
            
            self.model._unfreeze_PETF(args.unfrozen_modules)
            n_parameters_learn = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total number of trainable parameters of the model: ", n_parameters_learn)
                
        
        # initialize the full model from the checkpoint for inference.
        if args.pretrained_model_path:
            ckpt = torch.load(args.pretrained_model_path)
            self.model.load_state_dict(ckpt)#, strict = False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params= self.model.parameters(), lr= self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.max_epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        
        audio_loss, video_loss, audiovisual_loss = self.model(batch, is_trainval = True)
        train_loss = (audio_loss + video_loss + audiovisual_loss)/3
        
        batch_size = batch["tokens"].shape[0]
        
        self.log("loss", train_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("audio_loss", audio_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("video_loss", video_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("audiovisual_loss", audiovisual_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        
        batch_sizes = self.all_gather(batch_size)
        
        train_loss *= batch_sizes.size(0) / batch_sizes.sum()
        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))
        
        return train_loss
            
    def validation_step(self, batch, batch_idx):
        if self.args.is_matryoshka:
            val_audio_loss, val_video_loss, val_audiovisual_loss = self.model(batch, is_trainval = True, test_ratio_matry_audio = self.args.downsample_ratio_test_matry_audio, test_ratio_matry_video = self.args.downsample_ratio_test_matry_video)
        else:
            val_audio_loss, val_video_loss, val_audiovisual_loss = self.model(batch, is_trainval = True)
        val_loss = (val_audio_loss + val_video_loss + val_audiovisual_loss)/3
        
        batch_size = batch["tokens"].shape[0]
        
        self.log("loss_val", val_loss, batch_size=batch_size, sync_dist=True)
        self.log("loss_val_audio", val_audio_loss, batch_size=batch_size, sync_dist=True)
        self.log("loss_val_video", val_video_loss, batch_size=batch_size, sync_dist=True)
        self.log("loss_val_audiovisual", val_audiovisual_loss, batch_size=batch_size, sync_dist=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        if self.args.is_matryoshka:
            generated_ids = self.model(batch, is_trainval = False, modality = self.model.modality, test_ratio_matry_audio = self.args.downsample_ratio_test_matry_audio, test_ratio_matry_video = self.args.downsample_ratio_test_matry_video) if self.args.is_task_specific else self.model(batch, is_trainval = False, test_ratio_matry_audio = self.args.downsample_ratio_test_matry_audio, test_ratio_matry_video = self.args.downsample_ratio_test_matry_video)
        else:
            generated_ids = self.model(batch, is_trainval = False, modality = self.model.modality) if self.args.is_task_specific else self.model(batch, is_trainval = False)
        

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