#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:16:56 2025

@author: umbertocappellazzo
"""

import logging
import os
from argparse import ArgumentParser
import torch
import time
from avg_checkpoints_original import ensemble_original
from datamodule.data_module_LLM import DataModule_LLM
from lightning_MODROP import ModelModule_LLM

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy#, FSDPStrategy, DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger
#from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import time
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_trainer(args):
    seed_everything(args.seed, workers=True)  # Default seed: 42. Alternative: 7.
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=False,
        filename="{epoch}",
        save_top_k=args.num_check_save, 
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]
    
    find_unused_parameters_flag = False if args.modality == 'audio' else True

    return Trainer(
        precision='bf16-true',
        sync_batchnorm=True,
        num_sanity_val_steps=2,
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy= DDPStrategy(find_unused_parameters= find_unused_parameters_flag),  #FSDPStrategy(auto_wrap_policy=policy), #"deepspeed_stage_3" #fsdp" DDPStrategy(find_unused_parameters=False) DeepSpeedStrategy(logging_batch_size_per_gpu=4, stage=3, offload_optimizer=True, offload_parameters=True)
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        logger=WandbLogger(name=args.exp_name, project="AV_ASR_LLM", entity= "av_asr_llm"),
        gradient_clip_val=10.0,
        val_check_interval=args.val_check_interval,
        #accumulate_grad_batches=6
    )



def get_test_trainer(args):
    return Trainer(precision='bf16-true',
        num_nodes=1,
        devices=1,
        accelerator="gpu",
        logger=WandbLogger(name=args.exp_name, project="AV_ASR_LLM", entity= "av_asr_llm"),
    )

def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default="./results",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--root-dir",
        default="/ucappell/datasets", #   "/cappellazzo/avsr_llm/LRS3"   "/cappellazzo/AV_ASR/preprocessed_dataset"
        type=str,
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--exp-name",
        default= "", 
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--modality",
        default="audio",
        type=str,
        help="Type of input modality",
        choices=["audio", "video", "audiovisual", "audiovisual_avhubert"],
    )
    
    parser.add_argument(
        "--compression-mode",
        default= None,
        type= str,
        help= "How we compress the tokens.",
        choices= ["avg-pooling","stack", "resampler"]
    )
    

    parser.add_argument(
        "--llm-model",
        default= None,
        type=str,
        help="LLM model name",
        choices= ["TinyLlama/TinyLlama_v1.1", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-hf", 
                  "google/gemma-2b","google/gemma-2b", "google/gemma-2-9b",
                  "mistralai/Mistral-7B-v0.1", 
                  "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Llama-3.2-1B","meta-llama/Llama-3.2-3B",
                  "Qwen/Qwen2-1.5B", "Qwen/Qwen2-7B",
                  "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
    )
    parser.add_argument(
        "--hidden-size",
        default= 3072, #2048 3072 3584 4096 5120      Qwen2-1.5B: 1536. Qwen2-7B: 3584.
        type=int,
        help="Hidden size of the LLM.",
    )
    parser.add_argument(
        "--intermediate-size",
        default= 2048,
        type=int,
        help="Intermediate size of the projector.",
    )
    parser.add_argument(
        "--prompt-audio",
        default= "Transcribe speech to text.", #"Given the audio tokens, generate the corresponding text."
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-video",
        default= "Transcribe video to text.", #"Given the audio tokens, generate the corresponding text."
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-audiovisual",
        default= "Transcribe speech and video to text.", #"Given the audio tokens, generate the corresponding text."
        type=str,
        help="The prompt for the LLM.",
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-video-path",
        default= None,   # "/cappellazzo/avsr_llm/src/ckpt_pretrained/large_vox_iter5.pt",  "/cappellazzo/AV_ASR/autoavsr_v1.1/results/large_vox_iter5.pt", 
                                                                                #/cappellazzo/AV_ASR/autoavsr_v1.1/results/asr_trlibrispeech_base.pth", #asr_trlrs3vox2_base.pth vsr_trlrs2lrs3vox2avsp_base.pth raven_vox2lrs3_large_video.pth
        type=str,                                                               # vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_braven.pth
    )
    
    parser.add_argument(
        "--use-lora-avhubert",
        default = False,
        type = bool,
        help= "Whether to apply LoRA to the transformer module of AV-HuBERT."
        )
    parser.add_argument(
        "--audio-encoder-name",
        default = None, # "openai/whisper-medium.en/small.en/base.en/tiny.en/large",   "microsoft/wavlm-large", "av-hubert"
        type = str
        )
    parser.add_argument(
        "--unfrozen-modules",
        nargs="*",
        default= [None], #  "peft_llm","lora_avhubert"
        help="Which modules to train.",
        choices = [None, "embedding", "peft_llm","peft_vision", "lora_avhubert", "raven"]
    )
    parser.add_argument(
        "--add-PETF-LLM",
        default= None,
        type= str,
        help="Whether to add a PEFT module to the LLM.",
        choices= [None, "adapter", "lora","lora_peft"]
    )
    parser.add_argument(
        "--rank",
        default= 64,
        type=int,
        help="Rank for LoRA."
    )
    parser.add_argument(
        "--alpha",
        default= 8,
        type=int,
        help="Alpha for LoRA."
    )
    
    parser.add_argument(
        "--reduction-rate-adapter",
        default= 32,
        type=int,
        help="The reduction rate of the LLM PETF moduke. Set to None if add_PETF_LLM is False."
    )
    
    parser.add_argument(
        "--is-task-specific",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--probs",
        nargs="*",
        default=None, 
        type=float,
        help="Probabilities of choosing task.",
    )
    parser.add_argument(
        "--n-experts",
        default= 4,
        type= int,
        help = "The number of routed experts in MoE."
    )
    parser.add_argument(
        "--topk",
        default= 2,
        type= int,
        help= "The number of activated routed experts in MoE."
    )
    parser.add_argument(
        "--adapter-location",
        default = "FFN",
        choices = ["MHSA", "FFN", "LAYER"],
        type= str,
    )
    parser.add_argument(
        "--is-MoE",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--num-shared-experts",
        default = 1,
        type= int,
        help= "Number of shared experts. Default: 1."
    )
    parser.add_argument(
        "--apply-load-balancing-loss",
        default= False,
        type= bool,
        help= "Whether to apply the load balancing loss to the top-k router."
    )
    parser.add_argument(
        "--load-balancing-loss-coeff",
        default= 0.01,
        type= float,
    )
    parser.add_argument(
        "--is-hydra",
        default= False,
        type= bool,
        help= "Whether to share the downsample layer among the routed experts. The shared expert is not involved."
    )
    
    parser.add_argument(
        "--train-file",
        default="lrs3_train_transcript_lengths_seg16s_LLM_lowercase_greater25.csv", #"lrs3_30h_train_transcript_lengths_seg16s_LLM_lowercase_greater12.csv"  "lrs3_train_transcript_lengths_seg16s_LLM_lowercase_greater25.csv"
        type=str,
        help="Filename of training label list",
    )
    parser.add_argument(
        "--val-file",
        default="lrs3_test_transcript_lengths_seg16s_LLM_lowercase.csv",
        type=str,
        help="Filename of validation label list.",
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s_LLM_lowercase.csv",
        type=str,
        help="Filename of testing label list.",
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of machines used. (Default: 4)",
    )
    parser.add_argument(
        "--gpus",
        #nargs="*",
        default=1,  #[1,2,3,4,5]
        type=int,
        help="Number of gpus in each machine. (Default: 8)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default= None,
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of epochs for warmup. (Default: 5)",
    )
    parser.add_argument(
        "--max-epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--num-average-epochs",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--num-check-save",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--val-check-interval",
        default=1.,
    )
    parser.add_argument(
        "--downsample-ratio-audio",
        default=3,  #[1,2,3,4,5]
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-video",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    
    parser.add_argument(
        "--max-frames-audio",
        type=int,
        default=1000,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--max-frames-video",
        type=int,
        default=1500,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--max-frames-audiovisual",
        type=int,
        default=1000,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,   # 1e-3 for ASR and AVSR, 5e-4 for VSR.
        help="Learning rate. (Default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path of the checkpoint from which training is resumed.",
    )
    parser.add_argument(
        "--max-dec-tokens",
        default= 32,
        type=int,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-beams",
        default=15,
        type=int,
        help="Beams used for beam search",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=float,
        default=-1,
        help="Slurm job id",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default= 999999,  
        help="Level of signal-to-noise ratio (SNR)",
        choices= [999999,-5]
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    parser.add_argument(
        "--auto-test",
        default= True,
        help="Flag to use debug level for logging",
    )
    parser.add_argument(
        "--no-layernorm-projector",
        default=False,
        type=bool,
        help="Removes LayerNorm from the audio and video projectors",
    )
    
    
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    print(args)
    
    if args.slurm_job_id != -1:
        args.slurm_job_id = os.environ["SLURM_JOB_ID"]

    modelmodule = ModelModule_LLM(args)
    datamodule = DataModule_LLM(args, modelmodule.tokenizer, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.print(torch.cuda.memory_summary())
    
    if args.auto_test:
        
        args.pretrained_model_path = ensemble_original(args, args.num_average_epochs)
        #time.sleep(10)
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = get_test_trainer(args)
            #ckpt = torch.load(args.pretrained_model_path, map_location=lambda storage, loc: storage)
            #modelmodule.model.load_state_dict(ckpt)
            
            print("Evaluating on the ASR task!")
            args.modality = "audio"
            
            print("First evaluation round!")
            trainer.test(model=modelmodule, datamodule=datamodule)
            #print("Second evaluation round!")
            #trainer.test(model=modelmodule, datamodule=datamodule)
            #print("Third evaluation round!")
            #trainer.test(model=modelmodule, datamodule=datamodule)
            
            print("Evaluating on the VSR task!")
            args.modality = "video"
            
            print("First evaluation round!")
            trainer.test(model=modelmodule, datamodule=datamodule)
            print("Second evaluation round!")
            trainer.test(model=modelmodule, datamodule=datamodule)
            print("Third evaluation round!")
            trainer.test(model=modelmodule, datamodule=datamodule)
            
            print("Evaluating on the AVSR task!")
            args.modality = "audiovisual"
            
            print("First evaluation round!")
            trainer.test(model=modelmodule, datamodule=datamodule)
            #print("Second evaluation round!")
            #trainer.test(model=modelmodule, datamodule=datamodule)
            #print("Third evaluation round!")
            #trainer.test(model=modelmodule, datamodule=datamodule)
                    


if __name__ == "__main__":
    cli_main()
