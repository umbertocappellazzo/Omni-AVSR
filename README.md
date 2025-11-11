# 🎯 Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2024.12345-b31b1b.svg)](https://arxiv.org/abs/2511.07253)
[![Website](https://img.shields.io/badge/🌐-Website-blue.svg)](https://umbertocappellazzo.github.io/Omni-AVSR/)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=umbertocappellazzo.Omni-AVSR)](https://github.com/umbertocappellazzo/Omni-AVSR)
[![GitHub Stars](https://img.shields.io/github/stars/umbertocappellazzo/Omni-AVSR?style=social)](https://github.com/umbertocappellazzo/Omni-AVSR/stargazers)

**[Umberto Cappellazzo¹](#) · [Xubo Liu²](#) · [Pingchuan Ma¹](#) · [Stavros Petridis¹](#) · [Maja Pantic¹](#)**

¹Imperial College London ²University of Surrey

### 📄 [`Paper`](https://arxiv.org/abs/2511.07253) | 🌐 [`Project Page`](https://umbertocappellazzo.github.io/Omni-AVSR/) | 💻 [`Code`](https://github.com/umbertocappellazzo/Omni-AVSR) | 🔖 [`BibTeX`](#-citation)

</div>

---

## 📢 News

- **[11-2025]** 🚀 Code and models released!
- **[11-2025]** 📝 Paper submitted to arXiv.

---

## 🌟 Highlights

<div align="center">
  <img src="assets/pipeline.png" alt="Architecture" width="800"/>
  <p><i>Figure 1: Overall architecture of our proposed Omni-AVSR method.</i></p>
</div>

✨ **Key Contributions:**
- We present **Omni-AVSR**, the first audio-visual LLM that supports ASR, VSR, and AVSR jointly while enabling elastic inference under a single set of weights.
- Omni-AVSR hinges upon an optimized *matryoshka*-based 🪆 framework to support efficient multi-granularity training.
- To adapt the backbone LLM to all tasks in a parameter-efficient manner, Omni-AVSR uses three ad-hoc *LoRA*-based methods.
- Omni-AVSR achieves SoTA results on LRS2 and LRS3 benchmarks, whilst substantially reducing training and deployment costs.

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Main Results](#-main-results)
- [Setup](#-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Checkpoints](#-checkpoints)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

---

## 📝 Abstract

> **Omni-AVSR** is a unified multimodal large language model designed to perform auditory (ASR), visual (VSR), and audio-visual (AVSR) speech recognition within a single framework. It integrates **efficient Matryoshka-based multi-granularity training**, enabling flexible control of token compression rates for elastic inference across modalities. To adapt the LLM efficiently, Omni-AVSR introduces **Omni-LoRA**, offering shared, task-specific, and hybrid fine-tuning strategies. Experiments on LRS2 and LRS3 show that Omni-AVSR matches or surpasses state-of-the-art results while drastically reducing training and deployment costs.

---

## 🔬 Main Results

- ⚡ Omni-AVSR attains SoTA results on LRS2 and LRS3 while traning a single model (**Table 1**).
- ⚡ Omni-AVSR outperforms prior SoTA methods that support ASR-VSR-AVSR within a single model (**Table 2**).
- ⚡ Omni-AVSR achieves competitive WERs while requiring substantially fewer parameters and training data hours than all baselines (**Figure 2**).
- ⚡ Among several ablation studies, we report Omni-AVSR trend as we consider LLMs of different sizes from the Llama and Qwen 2.5 families (**Figure 3**). 


<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="assets/main_table.png" alt="Architecture" width="380"/>
        <p><i>Table 1: Main results on LRS2 and LRS3.</i></p>
      </td>
      <td align="center" width="50%">
        <img src="assets/unified_comparison.png" alt="Omni-LoRA Variants" width="380"/>
        <p><i>Table 2: Comparison with unified methods on LRS3.</i></p>
      </td>
    </tr>
  </table>
</div>


<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="assets/results.png" alt="Architecture" width="380"/>
        <p><i>Figure 2: Comparison with SoTA methods for AVSR on LRS3.</i></p>
      </td>
      <td align="center" width="50%">
        <img src="assets/scaling_LLM.png" alt="Omni-LoRA Variants" width="330"/>
        <p><i>Figure 3: Scaling trend of Omni-AVSR-ST when we increase the LLM size on LRS3..</i></p>
      </td>
    </tr>
  </table>
</div>

---

## 🛠 Setup 
Our setup follows that of [Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR).

### 1) Installation

Install necessary dependencies: 

```bash
   pip install -r requirements.txt
   cd av_hubert
   git submodule init
   git submodule update
   cd fairseq
   pip install --editable ./
```

### 2) Datasets Pre-processing

We rigorously follow auto-avsr [paper](https://arxiv.org/abs/2303.14307) to pre-process the LRS2 and LRS2 datasets. All the steps 
to achieve this can be found [here](https://github.com/mpc001/auto_avsr/tree/main/preparation).

For LRS3, the tree-structure of the directory is:

```text
LRS3  
└─── labels
     ├── lrs3_train_transcript.csv 
     ├── lrs3_test_transcript.csv 
     
└─── lrs3
     ├── lrs3_text_seg16s
     │    └── ...
     └── lrs3_video_seg16s
          └── ...
```

The label files in `[LRS3]/[labels]` and `[LRS2]/[labels]` undergo some processing to make them fit Omni-AVSR. For example, we lowercase the transcription and discard samples whose length is higher than a specific threshold to avoid training instability and peak GPU memory usage. Based on the desired training setting, the processed labels can be accessed below. Once downloaded, they must be moved to `[LRS3]/[labels]` or `[LRS2]/[labels]` subfolders. 

| Label Files | Dataset | Split| Hours |
|-----|:-----:|:-----:|:-----:|
|['lrs3_train_transcript.csv'](https://drive.google.com/file/d/1ahoVBZLl1j_LuAvplEWUdPpd4JpE5O8k/view?usp=drive_link)|LRS3|Train|433|
|['lrs2_train_transcript.csv'](https://drive.google.com/file/d/120YJQmVSdRvNHT-5qq9O0ESrfuPN5WqU/view?usp=drive_link)|LRS2|Train|225|
|['lrs3_test_transcript.csv'](https://drive.google.com/file/d/1DSeKQMUOJKNgE5wcvw91tc_CNsXgSa0l/view?usp=drive_link)|LRS3|Test|/|
|['lrs2_test_transcript.csv'](https://drive.google.com/file/d/1aBQqnTvBIgDxEdnQ0TIrfUuKXrFtF_b4/view?usp=drive_link)|LRS2|Test|/|

---

## 🎓 Training

### Preliminaries 
Before starting the training process, make sure you **1)** have a wandb account to track your experiments and **2)** have access to the pre-trained LLMs like Llama 3.2-1B (i.e., you need to request access from HF [here](https://huggingface.co/meta-llama/Llama-3.2-1B)). You also have to download the AV-HuBERT Large model pretrained on LRS3 + VoxCeleb2, accessible [here](https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt).

To set up the desired experiment to run, we have several main arguments to define, listed below:
<details open>
  <summary><strong>Main Arguments</strong></summary>
    
- `exp-dir`: Directory to save checkpoints and logs to.
- `root-dir`: Root directory of the preprocessed datasets.
- `wandb-project`: Name of the wandb project to track the results.
- `exp-name`: Experiment name. Location of checkpoints is `[exp_dir]`/`[exp_name]`.
- `modality`: The modality we use to train the methods. Choices: [`audio`, `video`, `audiovisual`].
- `llm-model`: The LLM backbone to use (e.g., `meta-llama/Llama-3.2-1B`).
- `compression-mode`: How to compress the audio and/or video tokens. Default: `avg-pooling`.
- `num-nodes`: Number of machines used. Default: `1`.
- `gpus`: Number of GPUs per machine. Default: `1`.
- `is-matryoshka`: Whether to use matryoshka representation learning.
- `auto-test`: Whether to run the evaluation stage right after the traning process is completed. Default: `True`.
- `is-task-specific`: True if we use a LoRA module for each tasks. This is set to True for *Omni-AVSR-T* and *Omni-AVSR-ST*.
- `use-shared-lora-task-specific`: True if we use a shared LoRA module. This is set to True for *Omni-AVSR-ST.
- `matry-weights`: The weight for the ASR/VSR/AVSR tasks (see Eq. 1 in our paper). Default: `[1,1,1]`.

</details>

There are **additional arguments** to define, which are mainly modality-specific. More details below.

<details>
  <summary><strong>Additional Arguments</strong></summary>
    
- `prompt-audio`: This is the prompt used for the ASR task. By default, this is set to `Transcribe speech to text.`. Likewise, we define the prompt for the VSR task (`prompt-video`) and AVSR task (`prompt-audiovisual`).
- `pretrain-avhubert-enc-video-path`: This is the path to the pre-trained AV-HuBERT video encoder.
- `audio-encoder-name`: The pre-trained audio encoder. Choices: [`openai/whisper-medium.en`, `microsoft/wavlm-large`, `av-hubert`].
- `unfrozen_modules`: The modules to unfroze before starting the training. This can be the LoRA modules of the LLM (`peft_llm`) or the LoRA modules of the video encoder (`lora_avhubert`). 
- `add_PETF_LLM`: Whether to fine-tune the LLM via LoRA. Set to `lora` if we use LoRA, else `None`.
- `reduction_lora` and `alpha`: if we fine-tune the LLM via LoRA, we need to define the factor by which we reduce the hidden size (`reduction_lora`) and the scaling factor (`alpha`). 
- `max-epochs`: Number of epochs to train Llama-AVSR.
- `num-average-epochs`: We average the last `num-average-epochs` ckpts. Default: `4`.
- `downsample-ratio-audio`: This argument defines the compression rate to apply to the audio tokens before the LLM. Likewise, we define the compression rate for the video tokens (`downsample-ratio-video`).
- `max-frames-audio`: Max number of audio frames in a batch. This number can be adjusted based on the own GPU memory. For video and audio-visual we define a similar value.  
- `lr`: The learning rate of the AdamW optimizer. For ASR and AVSR, we set it to `1e-3`, for VSR to `5e-4`.
- `weight-decay`: The weight decay of the optimizer. Default: `0.1`.
- `rank`: The rank of each LoRA module.

</details>

---

### Baselines

All the reported traning and evaluation commands pertains the LRS3 dataset. For the LRS2, you only need to update the train/eval/test labels files accordingly.

**Example 1: Llama-AVSR**
If you want to reproduce the results obtained by **Llama-AVSR**, we need to ru  the `train_LlamaAVSR.py` script. Suppose we want to train Llama-AVSR for the ASR task with an audio compression rate of 16:


```Shell
python train_LlamaAVSR.py --wandb-project wandb_project_name --root-dir path_to_root_dir --seed 7 \ 
--exp-name LRS3_audio_avg-pooling_Whisper-M_Llama3.2-1B_pool-16_LN_seed7 --modality audio --compression-mode avg-pooling \
--audio-encoder-name openai/whisper-medium.en --rank 32 --alpha 4 --llm-model meta-llama/Llama-3.2-1B \
--unfrozen-modules peft_llm --add-PETF-LLM lora --downsample-ratio-audio 16 --lr 1e-3 --max-frames-audio 1500 --gpus 2 \
--max-epochs 8 --num-average-epochs 1 --num-check-save 1 --train-file lrs3_train_transcript.csv \
--val-file lrs3_test_transcript.csv --test-file lrs3_test_transcript.csv
```

**Example 2: Llama-MTSK**

If you want to reproduce the results obtained by **Llama-MTSK**, we need to ru  the `train_LlamaAVSR.py` script. Suppose we want to train Llama-MTSK for the AVSR task with an audio compression rates [4,16] and video compression rates [2,5]:

```Shell
python train_LlamaAVSR.py --wandb-project wandb_project_name --root-dir path_to_root_dir --seed 7 \ 
--exp-name LRS3_audiovisual_Matry_avg-pool_AVH-L__Whisper-MLlama3.2-1B_pool-2-5_LN_seed42 --is-matryoshka True -modality audiovisual \
--compression-mode avg-pooling --pretrain-avhubert-enc-video-path path_to_avhubert_ckpt \
--audio-encoder-name openai/whisper-medium.en --rank 32 --alpha 4 --llm-model meta-llama/Llama-3.2-1B \
--unfrozen-modules peft_llm --add-PETF-LLM lora --downsample-ratio-audio 4 16 --downsample-ratio-video 2 5 \
--lr 1e-3 --max-frames-audiovisual 1500 --gpus 2 --max-epochs 8 --num-average-epochs 1 --num-check-save 1 \
--train-file lrs3_train_transcript.csv \ --val-file lrs3_test_transcript.csv --test-file lrs3_test_transcript.csv
```

**Example 3: Llama-MT**
If you want to reproduce the results obtained by **Llama-MT**, we need to ru  the `train_OmniAVSR.py` script. Suppose we want to train Llama-MT with an audio compression rates of 4 and video compression rate of 2:

```Shell
python train_OmniAVSR.py --wandb-project wandb_project_name --root-dir path_to_root_dir --seed 7 \ 
--exp-name LRS3_OmniAVSR_weights_1-15-1_avgpooling_Whisper-M_Llama3.2-1B_LoRA_pool-16-5_LN_seed7 \
--modality audiovisual --compression-mode avg-pooling --audio-encoder-name openai/whisper-medium.en \
--pretrain-avhubert-enc-video-path path_to_avhubert_ckpt --rank 32 --alpha 4 --llm-model meta-llama/Llama-3.2-1B \
--unfrozen-modules peft_llm lora_avhubert --use-lora-avhubert True --add-PETF-LLM lora --downsample-ratio-audio 4 \
--downsample-ratio-video 2 --lr 1e-3 --max-frames-audiovisual 1500 --gpus 2 --max-epochs 8 --num-average-epochs 1 \
--num-check-save 1 --matry-weights 1. 1.5 1. --is-single-matry-projector True
--train-file lrs3_train_transcript.csv \ --val-file lrs3_test_transcript.csv --test-file lrs3_test_transcript.csv
```

### Omni-AVSR

**Example 1: Omni-AVSR-S**

```Shell
python train_OmniAVSR.py --wandb-project wandb_project_name --root-dir path_to_root_dir --seed 7 \ 
--exp-name LRS3_OmniAVSR_Matry_weights_1-15-1_avg-pooling_Whisper-M_Llama3.2-1B_LoRA_pool-audio4-16_video2-5_LN_seed7/
--modality audiovisual --compression-mode avg-pooling --audio-encoder-name openai/whisper-medium.en --matry-weights 1. 1.5 1. \
--is-matryoshka True --pretrain-avhubert-enc-video-path path_to_avhubert_ckpt --rank 32 --alpha 4 \
--llm-model meta-llama/Llama-3.2-1B --unfrozen-modules peft_llm lora_avhubert --use-lora-avhubert True --add-PETF-LLM lora \
--downsample-ratio-audio 4 16 --downsample-ratio-video 2 5 --lr 1e-3 --max-frames-audiovisual 1500 --gpus 2 --max-epochs 8 \
--num-average-epochs 1 --num-check-save 1 --train-file lrs3_train_transcript.csv \
--val-file lrs3_test_transcript.csv --test-file lrs3_test_transcript.csv
```

**Example 2: Omni-AVSR-T**

```Shell
python train_OmniAVSR.py --wandb-project wandb_project_name --root-dir path_to_root_dir --seed 7 \ 
--exp-name LRS3_OmniAVSR_Matry_weights_1-15-1_avg-pooling_Whisper-M_Llama3.2-1B_LoRA_task-specific_pool-audio4-16_video2-5_LN_seed7/
--modality audiovisual --compression-mode avg-pooling --audio-encoder-name openai/whisper-medium.en --matry-weights 1. 1.5 1. --is-task-specific True\
--is-matryoshka True --pretrain-avhubert-enc-video-path path_to_avhubert_ckpt --rank 32 --alpha 4 \
--llm-model meta-llama/Llama-3.2-1B --unfrozen-modules peft_llm lora_avhubert --use-lora-avhubert True --add-PETF-LLM lora \
--downsample-ratio-audio 4 16 --downsample-ratio-video 2 5 --lr 1e-3 --max-frames-audiovisual 1500 --gpus 2 --max-epochs 8 \
--num-average-epochs 1 --num-check-save 1 --train-file lrs3_train_transcript.csv \
--val-file lrs3_test_transcript.csv --test-file lrs3_test_transcript.csv
```

**Example 3: Omni-AVSR-ST**

```Shell
python train_OmniAVSR.py --wandb-project wandb_project_name --root-dir path_to_root_dir --seed 7 \ 
--exp-name LRS3_OmniAVSR_Matry_weights_1-15-1_avg-pooling_Whisper-M_Llama3.2-1B_LoRA_task-specific_sharedLoRA_pool-audio4-16_video2-5_LN_seed7/
--modality audiovisual --compression-mode avg-pooling --audio-encoder-name openai/whisper-medium.en --matry-weights 1. 1.5 1. \
--is-task-specific True --use-shared-lora-task-specific True\
--is-matryoshka True --pretrain-avhubert-enc-video-path path_to_avhubert_ckpt --rank 32 --alpha 4 \
--llm-model meta-llama/Llama-3.2-1B --unfrozen-modules peft_llm lora_avhubert --use-lora-avhubert True --add-PETF-LLM lora \
--downsample-ratio-audio 4 16 --downsample-ratio-video 2 5 --lr 1e-3 --max-frames-audiovisual 1500 --gpus 2 --max-epochs 8 \
--num-average-epochs 1 --num-check-save 1 --train-file lrs3_train_transcript.csv \
--val-file lrs3_test_transcript.csv --test-file lrs3_test_transcript.csv
```



## 📈 Evaluation

### Evaluation on LRS2/LRS3 benchmarks

To test a trained model, either you set `--auto-test True` when starting a new training experiment, so the inference is performed automatically at the end of the traning, or you can run `eval_Llama-AVSR.py` for Llama-AVSR and Llama-MTSK or `eval_Omni-AVSR.py` for Llama-MT and Omni-AVSR. In both cases, a handful of inference arguments must be specified as follows:

<details open>
  <summary><strong>Inference Arguments</strong></summary>
    
- `max-dec-tokens`: Maximum number of tokens that can be generated by the LLM. Default: `32`.
- `num-beams`: Number of beams for beam search decoding. Default: `15`.
- `decode-snr-target`: Level of signal-to-noise ratio (SNR). Default: `999999` (i.e., clean setting).

</details>

If you want to run an inference experiment, you need to define the argument `--pretrained-model-path` and set it to the path to the pre-trained ckpt. Furthermore, you need to specify the same arguments used for the training. Below one example for the Omni-AVSR-ST that tests the pre-trained ckpt provided. By default, Omni-AVSR is evaluated on each compression rate and each task. If you want to test it only on one task, for example ASR, please set `--test-specific-modality True` and `--task-to-test audio`. If you want to test Omni-AVSR on a specific audio and/or video compression rate, set `--test-specific-ratio True` and set `--downsample-ratio-test-matry-audio`/`--downsample-ratio-test-matry-video` accordingly. 

```Shell
python eval_OmniAVSR.py --exp-name Omni_AVSR-ST_inference --pretrained-model-path path_to_ckpt --root-dir path_to_root_dir --modality audiovisual \
--compression-mode avg-pooling --audio-encoder-name openai/whisper-medium.en --pretrain-avhubert-enc-video-path path_to_avhubert_ckpt \
--llm-model meta-llama/Llama-3.2-1B --unfrozen-modules peft_llm lora_avhubert --use-lora-avhubert True --add-PETF-LLM lora \
--rank 32 --alpha 4 --downsample-ratio-audio 4 16 --downsample-ratio-video 2 5 --matry-weights 1. 1.5 1. --is-matryoshka True --is-task-specific True \
--use-shared-lora-task-specific True --test-file lrs3_test_transcript.csv --num-beams 15 --max-dec-tokens 32 --decode-snr-target 999999
```


### Evaluation on a custom video

If you fancy testing Omni-AVSR on a custom .mp4 video or you want to learn more about the pre-processing phase where we perform mouth cropping, please take a look in our dedicated [pre-processing section](./preprocessing_inference_example).

---

## 🎁 Checkpoints

We provide below three checkpoints, all of them use Llama 3.2-1B and ASR/VSR/AVSR weights = [1.0,1.5,1.0]. If you want to experiment with other checkpoint, please contact me.

| Model | Dataset | Link |
|-------|---------|----|
| LRS3_OmniAVSR_Matry_weights_1-15-1_avg-pooling_Whisper-M_Llama3.2-1B_LoRA_pool-audio4-16_video2-5_LN_seed7 | LRS3 | [Link](https://www.doc.ic.ac.uk/~ucappell/) |
| LRS3_OmniAVSR_Matry_weights_1-15-1_avg-pooling_Whisper-M_Llama3.2-1B_LoRA_task-specific_pool-audio4-16_video2-5_LN_seed7 | LRS3| [Link](https://www.doc.ic.ac.uk/~ucappell/) |
| LRS3_OmniAVSR_Matry_weights_1-15-1_avg-pooling_Whisper-M_Llama3.2-1B_LoRA_task-specific_sharedLoRA_pool-audio4-16_video2-5_LN_seed7 | LRS3| [Link](https://www.doc.ic.ac.uk/~ucappell/) |

---

## 🔖 Citation

If you find our work useful, please cite:

```bibtex
@article{cappellazzo2025Omni-AVSR,
  title={Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models},
  author={Umberto, Cappellazzo and Xubo, Liu and Pingchuan, Ma and Stavros, Petridis and Maja, Pantic},
  journal={arXiv preprint arXiv:2511.07253},
  year={2024}
}
```

---

## 🙏 Acknowledgements

- Our Code relies on [auto-avsr](https://github.com/mpc001/auto_avsr), [avhubert](https://github.com/facebookresearch/av_hubert), and [Llama-AVSR](https://github.com/umbertocappellazzo/Llama-AVSR) repositories
- Built with [PyTorch](https://pytorch.org/)

---

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Email: umbertocappellazzo@gmail.com
- Visit our [project page](https://umbertocappellazzo.github.io/Omni-AVSR/) and our [preprint](https://arxiv.org/abs/2511.07253)

---

<div align="center">
