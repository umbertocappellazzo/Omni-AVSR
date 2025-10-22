#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:10:25 2024

@author: umbertocappellazzo
"""

import logging
from argparse import ArgumentParser

from datamodule.data_module import DataModule_LLM
from lightning_OmniAVSR import ModelModule_LLM

from pytorch_lightning import Trainer
from Omni_AVSR.pytorch_lightning.loggers import WandbLogger

def get_trainer(args):
    return Trainer(precision='bf16-true',
                   num_nodes=1,
                   devices=1,
                   accelerator="gpu",
                   logger=WandbLogger(name=args.exp_name, project=args.wandb_project)
                   )
 
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-name",
        default= None,
        type=str,
        help="Experiment name.",
    )
    parser.add_argument(
        "--wandb-project",
        default= None, 
        type=str,
        help="wandb project name where to track the experiment metrics.",
    )
    parser.add_argument(
        "--modality",
        default="video",
        type=str,
        help="Type of input modality.",
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
        default= None,
        type=str,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--root-dir",
        default= None,
        type=str,
        help="Root directory of preprocessed dataset.",
    )
    parser.add_argument(
        "--is-task-specific",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--is-matryoshka",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--is-single-matry-projector",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--matry-weights",
        nargs="*",
        default=None, 
        type=float,
        help="Weights to apply to ASR, VSR, and AVSR tasks. If None, all weights are set to 1.",
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s_LLM_lowercase.csv",
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
        default= None,
        type=str,
        help="LLM model name",
    )
    parser.add_argument(
        "--audio-encoder-name",
        default = None,
        type = str
    )
    parser.add_argument(
        "--hidden-size",
        default= None,
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
        default= "Transcribe speech to text.",
        type=str,
        help="The audio prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-video",
        default= "Transcribe video to text.",
        type=str,
        help="The visual prompt for the LLM.",
    )
    parser.add_argument(
        "--prompt-audiovisual",
        default= "Transcribe speech and video to text.",
        type=str,
        help="The audiovisual prompt for the LLM.",
    )
    parser.add_argument(
        "--unfrozen-modules",
        nargs="*",
        default= [None],
        help="Which modules to train."
    )
    parser.add_argument(
        "--add-PETF-LLM",
        default= None,
        type= str,
        help="Whether to add a PEFT module to the LLM.",
        choices= [None, "lora"]
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
        "--downsample-ratio-audio",
        nargs="*",
        default=3,  #[1,2,3,4,5]
        type=int,
        help="Downsample audio ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-video",
        nargs="*",
        default=3,  #[1,2,3,4,5]
        type=int,
        help="Downsample video ratio.",
    )
    parser.add_argument(
        "--downsample-ratio-test-matry-audio",
        default=None,
        type=int,
        help="Downsample audio ratio for eval.",
    )
    parser.add_argument(
        "--downsample-ratio-test-matry-video",
        default=None,
        type=int,
        help="Downsample visual ratio for eval.",
    )
    parser.add_argument(
        "--max-dec-tokens",
        default= 32,
        type=int,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--num-beams",
        default= 15,
        type=int,
        help="Beams used for beam search.",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set.",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default= 999999,  
        help="Level of signal-to-noise ratio (SNR).",
    )
    parser.add_argument(
        "--no-layernorm-projector",
        default=False,
        type=bool,
        help="Removes LayerNorm from the audio and video projectors.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
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
        print("Evaluating on the ASR task!")
        args.modality = "audio"
        for rate_audio in args.downsample_ratio_audio:
            args.downsample_ratio_test_matry_audio = rate_audio
            print("First evaluation round, rate: ", rate_audio)
            trainer.test(model=modelmodule, datamodule=datamodule)
        
        print("Evaluating on the VSR task!")
        args.modality = "video"
        for rate_video in args.downsample_ratio_video:
            args.downsample_ratio_test_matry_video = rate_video
            print("First evaluation round, rate: ", rate_video)
            trainer.test(model=modelmodule, datamodule=datamodule)
            print("Second evaluation round, rate: ", rate_video)
            trainer.test(model=modelmodule, datamodule=datamodule)
            print("Third evaluation round, rate: ", rate_video)
            trainer.test(model=modelmodule, datamodule=datamodule)
        
        print("Evaluating on the AVSR task!")
        args.modality = "audiovisual"
        for rate_video in args.downsample_ratio_video:
            args.downsample_ratio_test_matry_video = rate_video
            for rate_audio in args.downsample_ratio_audio:
                args.downsample_ratio_test_matry_audio = rate_audio
                print(f"First evaluation round: audio rate {rate_audio}, video_rate {rate_video}.", rate_video)
                trainer.test(model=modelmodule, datamodule=datamodule)
    else:
    
        modelmodule.args.modality = "audio"
        
        print("First evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)
        
        modelmodule.args.modality = "video"
        
        print("First evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)
        print("Second evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)
        print("Third evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)
        
        modelmodule.args.modality = "audiovisual"
        
        print("First evaluation round!")
        trainer.test(model=modelmodule, datamodule=datamodule)

if __name__ == "__main__":
    cli_main()