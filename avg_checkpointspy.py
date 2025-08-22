#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:01:38 2024

@author: umbertocappellazzo
"""

import os

import torch


def average_checkpoints(last):
    avg = None
    for path in last:
        states = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


exp_dir = "/ucappell/AVSR-LLMs/results/LRS3_audiovisual_stack_AVH-Large_Whisper-M_Llama3.2-1B_pool-4-2_LN_seed7"
names = ["epoch=6.ckpt","epoch=7.ckpt"]
last = [os.path.join(exp_dir, name) for name in names 
    ]

model_path = os.path.join(exp_dir, "model_avg_2.pth")

torch.save(average_checkpoints(last), model_path)