import json
from typing import Dict
import urllib

import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
  CenterCropVideo,
  NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
  ApplyTransformToKey,
  ShortSideScale,
  UniformTemporalSubsample,
  UniformCropVideo
)

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second


class PackPathway(torch.nn.Module):
  """
  Transform for converting video frames as a list of tensors.
  """

  def __init__(self):
    super().__init__()

  def forward(self, frames: torch.Tensor):
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
      frames,
      1,
      torch.linspace(
        0, frames.shape[1]-1, frames.shape[1]//slowfast_alpha
      ).long(),
    )
    frame_list = [slow_pathway, fast_pathway]
    return frame_list


transform = ApplyTransformToKey(
  key='video',
  transform=Compose(
    [
      UniformTemporalSubsample(num_frames),
      Lambda(lambda x: x/255.0),
      NormalizeVideo(mean, std),
      ShortSideScale(
        size=side_size
      ),
      CenterCropVideo(crop_size),
      PackPathway()
    ]
  ),
)


def main():
  device = 'cuda'

  model_name = 'slowfast_r50'
  model = torch.hub.load('facebookresearch/pytorchvideo', model=model_name, pretrained=True)

  model = model.to(device)
  model = model.eval()

  # Download the id to label mapping for the Kinetics 400 dataset
  json_url = 'https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json'
  json_filename = 'kinetics_classnames.json'
  try:
    urllib.URLopener().retrieve(json_url, json_filename)
  except:
    urllib.request.urlretrieve(json_url, json_filename)

  # Create an id to label name mapping
  with open(json_filename, 'r') as f:
    kinetics_classnames = json.load(f)

  kinetics_id_to_classname = {}
  for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', '')

  # Download an example video
  url_link = 'https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4'
  video_path = 'archery.mp4'
  try:
    urllib.URLopener().retrieve(url_link, video_path)
  except:
    urllib.request.urlretrieve(url_link, video_path)

  # Load the video and transform it to the input format required by the model
  # Select the duration of the clip to load by specifying the start and end duration
  # The start_sec should correspond to where the action occurs in the video
  start_sec = 0
  end_sec = start_sec+clip_duration

  # Initialize an EncodedVideo helper class and load the video
  video = EncodedVideo.from_path(video_path)

  # Load the desired clip
  video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

  # Apply a transform to normalize the video input
  video_data = transform(video_data)

  # Move the inputs to the desired device
  inputs = video_data["video"]
  inputs = [i.to(device)[None, ...] for i in inputs]

  # Get predictions
  # Pass the input clip through the model
  preds = model(inputs)

  # Get the predicted classes
  post_act = torch.nn.Softmax(dim=1)
  preds = post_act(preds)
  pred_classes = preds.topk(k=5).indices[0]

  # Map the predicted classes to the label names
  pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
  print(f'Top 5 predicted labels: {", ".join(pred_class_names)}')


if __name__ == '__main__':
  main()
