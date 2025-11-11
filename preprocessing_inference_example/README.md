# Pre-processing + Inference for a custom video clip

---
We present here an example of how to pre-process a video clip and generate its transcription using Omni-AVSR-ST.

## Packages Installation

Compared to the environment we are going to use for the inference (see the README file in the main page), we need to adjust the packages as below due to some conflicts that arise when using the RetinaFace detector:

```Shell
pip install scikit-image
pip install numpy==1.24.4
pip install av==13.0.0
```

## RetinaFace installation

To preprocess our video clip, we need the RetinaFace detector. Please, follow the next steps to install it. You can choose to install `ibug.face_detection` and `ibug.face_alignment` in your preferred location. 


### `ibug.face_detection` 
Clone the repo:
```Shell
git clone https://github.com/hhj1897/face_detection.git
```
Since the size of the pre-trained model in the repo is larger than 100MB, the model is not cloned. To obviate this, we might use [Git LFS](https://git-lfs.github.com/). However, it seems like there is an issue with the ibug.face_detector repo, look [here](https://github.com/hhj1897/face_detection/issues/4). Therefore, one workaround is to download the model [Resnet50_final.pth](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1) and move it inside the repo at: `face_detection/ibug/face_detection/retina_face/weights`. Once this is done, let's do this:
```Shell
cd face_detection
pip install -e .
```

### `ibug.face_alignment`
```Shell
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
```
At this point, the RetinaFace detector is installed and we are ready to extract the mouth regions of interest (RoIs).

## Pre-processing of the video clip

The pre-processing stage creates the cropped mouth from a raw video, which includes face detection, landmarks tracking, face transformation and mouth cropping. 

The preprocessing script requires the definition of the path to the input video we want to preprocess (`path_to_input_video`). It is optional to provide the transcription of the video (`gold_transcription`). In this folder we include a video from the LRS3 testset, `video_example.mp4`, so you can try with this one if you want.

```Shell
python preprocess_video.py --path_to_input_video path_to_input_video --gold_transcription gold_transcription
```

The script returns the preprocessed video, the audio file in wav format and a csv file for the necessary information to perform the inference (`test_file.csv`).

## Inference Stage

Before performing the inference, we need to create a folder. The default name is `inference_test`, but you can change it inside the `preprocess_video.py` script. Once it's created, move the preprocessed video and audio files from the previous step. Also, remember to switch back to the environment defined in the README file in the root directory.

To run the inference with Omni-AVSR-ST using both audio and video (you can also test with only audio or only video), we navigate back to the parent directory `Omni-AVSR` and run the following:

```Shell
python eval_OmniAVSR.py --exp-name Omni_AVSR-ST_inference_example --pretrained-model-path path_to_ckpt --root-dir path_to_root_dir --modality audiovisual \
--compression-mode avg-pooling --audio-encoder-name openai/whisper-medium.en --pretrain-avhubert-enc-video-path path_to_avhubert_ckpt \
--llm-model meta-llama/Llama-3.2-1B --unfrozen-modules peft_llm lora_avhubert --use-lora-avhubert True --add-PETF-LLM lora \
--rank 32 --alpha 4 --downsample-ratio-audio 4 16 --downsample-ratio-video 2 5 --matry-weights 1. 1.5 1. --is-matryoshka True --is-task-specific True \
--use-shared-lora-task-specific True --test-file test_file.csv --num-beams 15 --max-dec-tokens 32 --decode-snr-target 999999
```

where `--root-dir` points to the path of the folder `inference_test`.

## Acknowledgements 

The pre-processing step highly relies on [auto-avsr](https://github.com/mpc001/auto_avsr). 
