import torch
import torchvision
import os
from data_module import AVSRDataLoader
import soundfile as sf

class AVSRDataLoader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from detector import LandmarksDetector
        from video_process import VideoProcess
        self.landmarks_detector = LandmarksDetector(device="cuda:0")
        self.video_process = VideoProcess(convert_gray=False)

    def forward(self, data_filename):
        video = self.load_video(data_filename)
        landmarks = self.landmarks_detector(video)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        return video

    def load_video(self, data_filename):
        return torchvision.io.read_video(data_filename, pts_unit="sec")[0].numpy()

def save2vid(filename, vid, frames_per_second):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchvision.io.write_video(filename, vid, frames_per_second)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess videos and save them.")
    parser.add_argument("--path_to_input_video", type=str, required=True, help="Input video to process.")
    parser.add_argument("--gold_transcription", type=str, default = None, help="The text transcription of the video. Otherwise it is set to NA.")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    video_dataloader = AVSRDataLoader()
    preprocessed_video = video_dataloader(args.path_to_input_video)

    save2vid(args.path_to_input_video.split(".")[0]+"_preprocessed.mp4", preprocessed_video, frames_per_second=30)

    _, audio, info = torchvision.io.read_video(args.path_to_input_video, pts_unit="sec")

    audio_np = audio.numpy().T


    sf.write(args.path_to_input_video.split(".")[0]+"_preprocessed.wav", audio_np, info['audio_fps'])

    transcription = args.gold_transcription if args.gold_transcription else "NA" 
    row_to_write = ["inference_test",args.path_to_input_video.split(".")[0]+"_preprocessed.mp4",0,0,transcription]
    
    with open("test_file.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row_to_write)
