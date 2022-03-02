from fractions import Fraction
import random
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/clip_sampling.py
class RandomClipSampler(ClipSampler):
  def __call__(self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]) -> ClipInfo:
    max_possible_clip_start = max(video_duration-self._clip_duration, 0)
    clip_start_sec = Fraction(random.uniform(0, max_possible_clip_start))
    clip_end_sec = clip_start_sec+self._clip_duration
    time = float((clip_start_sec+clip_end_sec)/2)
    return ClipInfo(clip_start_sec, clip_end_sec, time, 0, True)
