#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:14:11 2024

@author: umbertocappellazzo
"""

import os
import torch
import torchaudio
import torchvision

def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data

def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    #return waveform[0,:].unsqueeze(0).transpose(1, 0)
    return waveform.transpose(1,0)

class AVDataset_LLM(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        downsample_ratio = None,
        is_matryoshka = False
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        
        
        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]
        
        if (modality == "video" or modality == "audiovisual"):
            if is_matryoshka:
                self.downsample_video = None
            else:
                self.downsample_video = downsample_ratio if downsample_ratio != 1 else None 
        
    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, _, text = path_count_label.split(",")
            paths_counts_labels.append((dataset_name, rel_path, input_length, text))
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, _, text = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            
            if self.downsample_video:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
            
            return {"video": video, "tokens": text}
        elif self.modality == "audio":
            audio = load_audio(path)
            
            audio = self.audio_transform(audio)
                  
            return {"audio": audio, "tokens": text}
        elif self.modality == "audiovisual":
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            
            if self.downsample_video:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
                
            return {"video": video, "audio": audio, "tokens": text}
        

    def __len__(self):
        return len(self.list)