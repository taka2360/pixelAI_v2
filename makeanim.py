import os
import numpy as np
import moviepy.editor as mpy
from IPython.display import display

os.environ["SDL_VIDEODRIVER"] = "dummy"
VIDEO_MAXDURATION_SEC = 60

def make_anim(images, fps=60):
    duration = len(images) / fps
    if VIDEO_MAXDURATION_SEC < duration: # 長すぎるなら
        images = images[:(fps * VIDEO_MAXDURATION_SEC)]
        duration = VIDEO_MAXDURATION_SEC # 打ち切る
        print(f'truncated: len={len(images)}, duration={duration}')

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        return x.swapaxes(0, 1).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip