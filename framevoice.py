## Copyright (C) 2023 Andreas Bertsatos <abertsatos@biol.uoa.gr>
##
## This file is part of the statistics package for GNU Octave.
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## Input arguments
## audio   audio signal for clipping
## fs      sampling frequency
## frame   frame length in ms
## step    sliding step in ms

import os
import cv2
import math
import argparse
import numpy as np
from typing import NamedTuple

class FrameVoiceIdx(NamedTuple):
    err: int    # audio signal cliped correctly = 0, otherwise 1
    beg: int    # starting time stamp
    end: int    # ending time stamp

def framevoice (audio, fs, frame, step) -> FrameVoiceIdx:
    ## Get length of audio stream
    audio_length = audio.shape[0]
    ## Set sliding window in ms
    frame_length = math.floor(frame * fs / 1000);
    sliding_step = math.floor(step * fs / 1000);

    ## Slide through the audio signal and calculate moving RNMS
    N = 1 / frame_length
    #i = 0
    aRMS = np.array([])
    for idx in range(0,audio_length - frame_length,sliding_step):
        ## Get frames starting and ending indices
        frame_beg = idx
        frame_end = idx + frame_length
        ## Extract audio frame
        audio_seg = audio[frame_beg:frame_end]
        ## Calculate RMS
        RMS = math.sqrt(N * np.sum(np.multiply(audio_seg, audio_seg)))
        aRMS = np.append(aRMS, RMS)
        #i = i + 1
    offset_beg = frame_length * 4
    clip_beg = (np.argmax(aRMS > 0.005) + 1) * sliding_step - offset_beg
    err = 0
    if (clip_beg < 1):
        clip_beg = 1
        err = 1
    aRMS = np.flip(aRMS)
    offset_end = frame_length * 2
    clip_end = (aRMS.shape[0] - np.argmax(aRMS > 0.005)) * sliding_step + offset_end
    if (clip_end > audio_length):
        clip_end == audio_length
        err = 1
    FrameVoiceIdx.err = err
    FrameVoiceIdx.beg = clip_beg
    FrameVoiceIdx.end = clip_end
    return FrameVoiceIdx
