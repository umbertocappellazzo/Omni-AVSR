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
from utils.avg_checkpoints import ensemble_original
from datamodule.data_module import DataModule_LLM
from Omni_AVSR.lightning_OmniAVSR import ModelModule_LLM

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy#, FSDPStrategy, DeepSpeedStrategy
from pytorch_lightning.loggers import WandbLogger

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
        logger=WandbLogger(name=args.exp_name, project=args.wandb_project),
        gradient_clip_val=10.0,
        val_check_interval=args.val_check_interval,
        #accumulate_grad_batches=
    )



def get_test_trainer(args):
    return Trainer(precision='bf16-true',
        num_nodes=1,
        devices=1,
        accelerator="gpu",
        logger=WandbLogger(name=args.exp_name, project=args.wandb_project),
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
        default=None,
        type=str,
        help="Root directory of preprocessed dataset.",
    )
    parser.add_argument(
        "--wandb-project",
        default= None, 
        type=str,
        help="wandb project name where to track the experiment metrics.",
    )
    parser.add_argument(
        "--seed",
        default=7,
        type=int,
    )
    parser.add_argument(
        "--exp-name",
        default= "", 
        type=str,
        help="Experiment name.",
    )
    parser.add_argument(
        "--modality",
        default="audio",
        type=str,
        help="Type of input modality.",
        choices=["audio", "video", "audiovisual", "audiovisual_avhubert"],
    )
    
    parser.add_argument(
        "--compression-mode",
        default= None,
        type= str,
        help= "How we compress the tokens.",
        choices= ["avg-pooling","stack"]
    )
    parser.add_argument(
        "--llm-model",
        default= None,
        type=str,
        help="LLM model name.",
        choices= ["TinyLlama/TinyLlama_v1.1", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-hf",
                  "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3.1-8B", "meta-llama/Llama-3.2-1B","meta-llama/Llama-3.2-3B",
                  "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B"]
    )
    parser.add_argument(
        "--hidden-size",
        default= None, #2048 3072 3584 4096 5120
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
        help="The audio-visual prompt for the LLM.",
    )
    parser.add_argument(
        "--pretrain-avhubert-enc-video-path",
        default= None,
        type=str,
    )
    parser.add_argument(
        "--use-lora-avhubert",
        default = False,
        type = bool,
        help= "Whether to apply LoRA to the transformer module of AV-HuBERT."
    )
    parser.add_argument(
        "--audio-encoder-name",
        default = None, # "openai/whisper-medium.en/small.en/base.en/tiny.en/large",   "microsoft/wavlm-large"
        type = str
    )
    parser.add_argument(
        "--unfrozen-modules",
        nargs="*",
        default= [None], #  "peft_llm","lora_avhubert"
        help="Which modules to train.",
        choices = [None, "peft_llm", "lora_avhubert"]
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
        "--is-task-specific",
        default= False,
        type= bool,
    )
    parser.add_argument(
        "--use-shared-lora-task-specific",
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
        help="Whether to use a single projector for multiple compression rates when Matryosha is used. This is set to True when Matryoshka is not used."
    )
    parser.add_argument(
        "--matry-weights",
        nargs="*",
        default=None, 
        type=float,
        help="Weights to apply to ASR, VSR, and AVSR tasks. If None, all weights are set to 1.",
    )
    parser.add_argument(
        "--train-file",
        default="lrs3_train_transcript_lengths_seg16s_LLM_lowercase_greater25.csv",
        type=str,
        help="Filename of training label list.",
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
        help="Number of machines used.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of gpus in each machine.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default= None,
        type=str,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of epochs for warmup.",
    )
    parser.add_argument(
        "--max-epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--num-average-epochs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num-check-save",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--val-check-interval",
        default=1.,
    )
    parser.add_argument(
        "--downsample-ratio-audio",
        nargs="*",
        default=3,
        type=int,
        help="Downsample ratio for audio.",
    )
    parser.add_argument(
        "--downsample-ratio-video",
        nargs="*",
        default=3,
        type=int,
        help="Downsample ratio for video.",
    )
    parser.add_argument(
        "--downsample-ratio-test-matry-audio",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--downsample-ratio-test-matry-video",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--max-frames-audio",
        type=int,
        default=1000,
        help="Maximal number of audio frames in a batch. This must be specificied when the ASR task is considered.",
    )
    parser.add_argument(
        "--max-frames-video",
        type=int,
        default=1500,
        help="Maximal number of video frames in a batch. This must be specificied when the VSR task is considered.",
    )
    parser.add_argument(
        "--max-frames-audiovisual",
        type=int,
        default=1000,
        help="Maximal number of audiovisual frames in a batch. This must be specificied when the AVSR task is considered.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,   # 1e-3 for ASR and AVSR, 5e-4 for VSR.
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay.",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set.",
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
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--num-beams",
        default=15,
        type=int,
        help="Beams used for beam search.",
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
    )
    parser.add_argument(
        "--auto-test",
        default= True,
        help="Whether to test the model after traning within the same run.",
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
        help="Removes LayerNorm from the audio and video projectors.",
    )
    
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    
    print(args)
    
    if args.slurm_job_id != -1:
        args.slurm_job_id = os.environ["SLURM_JOB_ID"]
    
    if not args.is_matryoshka:
        if type(args.downsample_ratio_audio) == list:
            args.downsample_ratio_audio = args.downsample_ratio_audio[0]
        if type(args.downsample_ratio_video) == list:
            args.downsample_ratio_video = args.downsample_ratio_video[0]
    
    modelmodule = ModelModule_LLM(args)
    datamodule = DataModule_LLM(args, modelmodule.tokenizer, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
    trainer.print(torch.cuda.memory_summary())
    
    if args.auto_test:
        
        args.pretrained_model_path = ensemble_original(args, args.num_average_epochs)
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = get_test_trainer(args)
        
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
                # Llama-MT.
                print("Evaluating on the ASR task!")
                args.modality = "audio"
                
                print("First evaluation round!")
                trainer.test(model=modelmodule, datamodule=datamodule)
            
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

if __name__ == "__main__":
    cli_main()
