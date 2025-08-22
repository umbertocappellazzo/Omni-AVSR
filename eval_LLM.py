#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:10:25 2024

@author: umbertocappellazzo
"""

import logging
from argparse import ArgumentParser

from datamodule.data_module_LLM import DataModule_LLM
from pytorch_lightning import Trainer
from lightning_LLM import ModelModule_LLM
from pytorch_lightning.loggers import WandbLogger
import time
import datetime
# Set environment variables and logger level
#logging.basicConfig(level=logging.WARNING)


def get_trainer(args):
    return Trainer(precision='bf16-true',
                   num_nodes=1,
                   devices=1,
                   accelerator="gpu",
                   logger=WandbLogger(name=args.exp_name, project="AV_ASR_LLM", entity= "av_asr_llm")
                   )
 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-name",
        default= None,
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--modality",
        default="video",
        type=str,
        help="Type of input modality",
        choices=["audio", "video", "audiovisual"],
    )

    parser.add_argument(
        "--compression-mode",
        default= None,
        type= str,
        help= "How we compress the tokens.",
        choices= ["avg-pooling","stack", "resampler"]
    )

    parser.add_argument(
        "--pretrained-model-path",                      
        default= None, #"/cappellazzo/AVSR-LLMs_further/   #/cappellazzo/AV_ASR/autoavsr_v1.1/results
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--root-dir",
        default="/checkpoints/hongliechen/autoavsr",  # "/cappellazzo/avsr_llm/LRS3"  "/cappellazzo/AV_ASR/preprocessed_dataset"
        type=str,
        help="Root directory of preprocessed dataset",
    )
    
    parser.add_argument(
        "--is-matryoshka",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--is-MoE",
        default= False,
        type= bool,
    )
    
    parser.add_argument(
        "--matry-weights",
        nargs="*",
        default=None,  #[1,2,3,4,5]
        type=float,
        help="Weights to apply to different Matryoshka features. If None, all weights are set to 1.",
    )
    
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s_LLM_lowercase.csv", #lrs3_test_transcript_lengths_seg16s_LLM_lowercase.csv  wild_test_transcript_lengths_seg16s_LLM_lower.csv
        type=str,
        help="Filename of testing label list.",
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-video-path",
        default= None, #"/cappellazzo/AV_ASR/autoavsr_v1.1/results/large_vox_iter5.pt", "/cappellazzo/avsr_llm/src/ckpt_pretrained/large_vox_iter5.pt"
        type=str,                                                               
    )
    parser.add_argument(
        "--use-lora-avhubert",
        default = False,
        type = bool,
        help= "Whether to apply LoRA to the transformer module of AV-HuBERT."
        )
    parser.add_argument(
        "--llm-model",
        default= "meta-llama/Llama-3.2-3B",
        type=str,
        help="LLM model name",
    )
    parser.add_argument(
        "--audio-encoder-name",
        default = "openai/whisper-small.en", # "openai/whisper-medium.en",  "microsoft/wavlm-large"
        type = str
        )
    
    parser.add_argument(
        "--hidden-size",
        default= 3072, # 2048 3072 4096 5120
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
        "--unfrozen-modules",
        nargs="*",
        default= [None],  #"peft_llm","lora_avhubert"
        help="Which modules to train."
    )
    parser.add_argument(
        "--add-PETF-LLM",
        default= None,
        type= str,
        help="Whether to add a PEFT module to the LLM.",
        choices= [None, "adapter", "lora","lora_peft", "omni-smola-multimodal", "omni-smola-multi-unimodal"]
    )
    
    parser.add_argument(
        "--reduction-rate-adapter",
        default= 32,
        type=int,
        help="The reduction rate of the LLM PETF moduke. Set to None if add_PETF_LLM is False."
    )
    parser.add_argument(
        "--location_adapter",
        default= "FFN",
        type=str,
        help="Whether to place the PETF module of the LLM in the MHSA/FFN layer."
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
        "--reduction_rate_PETF_LLM",
        default= None,
        type=int,
        help="The reduction rate of the LLM PETF moduke. Set to None if add_PETF_LLM is False."
    )
    
    parser.add_argument(
        "--downsample-ratio-audio",
        nargs="*",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-test-matry",
        default=None,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-video",
        nargs="*",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-audiovisual",
        default=3,
        type=int,
        help="Downsample ratio.",
    )
    parser.add_argument(
        "--max-dec-tokens",
        default= 32,
        type=int,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-beams",
        default= 15,
        type=int,
        help="Beams used for beam search",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default= 999999,  
        help="Level of signal-to-noise ratio (SNR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
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
    init_logger(args.debug)
    
    if not args.is_matryoshka:
        if type(args.downsample_ratio_audio) == list:
            args.downsample_ratio_audio = args.downsample_ratio_audio[0]
        if type(args.downsample_ratio_video) == list:
            args.downsample_ratio_video = args.downsample_ratio_video[0]
            
    modelmodule = ModelModule_LLM(args)
    datamodule = DataModule_LLM(args, modelmodule.tokenizer, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)
    
    if args.is_matryoshka:
        if args.modality == "audio":
            for rate_audio in args.downsample_ratio_audio:
                args.downsample_ratio_test_matry = rate_audio
                print("First evaluation round, rate: ", rate_audio)
                trainer.test(model=modelmodule, datamodule=datamodule)
        elif args.modality == "video":
            for rate_video in args.downsample_ratio_video:
                args.downsample_ratio_test_matry = rate_video
                print("First evaluation round, rate: ", rate_video)
                trainer.test(model=modelmodule, datamodule=datamodule)
                print("Second evaluation round, rate: ", rate_video)
                trainer.test(model=modelmodule, datamodule=datamodule)
                print("Third evaluation round, rate: ", rate_video)
                trainer.test(model=modelmodule, datamodule=datamodule)
        else:
            for rate_video in args.downsample_ratio_video:
                for rate_audio in args.downsample_ratio_audio:
                    args.downsample_ratio_test_matry = [rate_video, rate_audio]
                    print(f"First evaluation round: audio rate {rate_audio}, video_rate {rate_video}.", rate_video)
                    trainer.test(model=modelmodule, datamodule=datamodule)
        
        
    else:
        print("First evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)
        print("Second evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)
        print("Third evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)
    


if __name__ == "__main__":
    cli_main()