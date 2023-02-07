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

## A python script from batch processing video clips to extract voice segments
## and their corresponding cartoonified frames. Useful information about the
## proccessing of each video clip is stored in a CSV file named "processed.csv"
## which can be used for creating audio-synced sprites in Unity. The expected
## sprites are saved in the respective folder for inspection.

import os
import cv2 
import math
import glob
import librosa
import moviepy
import numpy as np
import pandas as pd
import noisereduce as nr
from ffmpy import FFmpeg
from moviepy.editor import *
from natsort import natsorted
from scipy.io import wavfile
from inference import Inference
from framevoice import framevoice
from framevoice import FrameVoiceIdx
from stripframes import video2frames

## Set video frame parameters
vfr = 25    # framerate of the videos
sfr = 5     # frames to skip from each video during frame extraction
ifr = 10    # initial frames to ignore from the start of the video
beg = ifr / vfr

## Declare folders
video_path = "videos"       # original videos
audio_path = "voice"        # folder for stripped audio files
frame_path = "frames"       # extracted frames from video
toons_path = "toons"        # cartoonized frames
sprit_path = "sprites"      # cartoonized sprites with synced voice

## Declare filename for storing processing information
output_csv = "processed.csv"

## Declare inference parameters
model_path="saved_models/style3.pth"        # see folder for alternatives
yolov_path="saved_models/yolov5s-face.onnx"
device = "cuda"
infer = Inference(model_path, yolov_path, device)

## Ignore warning for overflow in inference model
import warnings
warnings.filterwarnings("ignore")

## Get list of available video files (sorted numerically in ascending order)
files = glob.glob(video_path+os.sep+"*.mp4")
files = natsorted(files)

## Prepare DataFrame for storing information of processed videos
header = ["video","duration","voice", "starting frame", "ending frame",
        "frames", "starting time", "ending time", "length", "processed"]
Tindex = range(1,files.__len__()+1)
tbl = pd.DataFrame(columns=header, index=Tindex)
idx = 1
## Process each video
for filename in files:
    ## Add video filename to dataframe
    tbl.loc[idx,"video"] = filename.replace(video_path+os.sep, "")
    ## Extract audio stream
    audioname = filename.replace(".mp4", ".wav")
    audioname = audioname.replace(video_path, audio_path)
    ff = FFmpeg(
        global_options = {"-y -hide_banner -loglevel error"},
        inputs = {filename: None},
        outputs = {audioname: ["-vn", "-acodec", "pcm_s16le"]}
    )
    ff.run()
    print("Extracted audio stream to: "+audioname)
    ## Read audio stream
    samplerate = librosa.get_samplerate(audioname)
    print("Sampling rate at: {}Hz".format(samplerate))
    audiodata, samplerate = librosa.load(audioname, offset=beg, sr=samplerate)
    duration = librosa.get_duration(y=audiodata, sr=samplerate)
    print("Duration: {:.3f}s".format(duration+beg))
    ## Apply noise reduction
    reduced_noise = nr.reduce_noise(y=audiodata, sr=samplerate)
    ## Find starting time stamp of voice in audio stream
    FrameVoiceIdx = framevoice(reduced_noise, samplerate, 100, 25)
    if FrameVoiceIdx.err == 1:
        print("Audio file {} clipped inappropriately".format(audioname))
        tbl.loc[idx,"processed"] = "error"
    ## Calculate time stamps and frames
    start_time = FrameVoiceIdx.beg / samplerate     # in seconds
    stop_time = FrameVoiceIdx.end / samplerate      # in seconds
    frame_beg = math.floor (vfr * start_time) + 1
    frame_end = math.floor (vfr * stop_time)
    clip_beg = round ((frame_beg - 1) / vfr * samplerate)
    ## Find last frame and add extra audio time to keep frame intervals fixed
    frame_id = range(frame_beg, frame_end, sfr)[-1]
    extended = frame_id + sfr - 1
    clip_end = round (extended / vfr * samplerate)
    if (clip_end > audiodata.shape[0]):
        clip_end = audiodata.shape[0]
        if FrameVoiceIdx.err == 0:
            print("Audio file {} clipped inappropriately".format(audioname))
            tbl.loc[idx,"processed"] = "error"
    else:
        print("Audio file {} processed OK".format(audioname))
        tbl.loc[idx,"processed"] = "OK"
    ## Clip denoised audio and save to file
    clipped_audio = reduced_noise[clip_beg:clip_end]
    audiolength = clipped_audio.shape[0] / samplerate
    print("Clipped audio is {}s".format(audiolength))
    wavfile.write(audioname, samplerate, clipped_audio)
    ## Add records to dataframe
    tbl.loc[idx,"duration"] = duration+beg
    tbl.loc[idx,"striped audio"] = audioname.replace(audio_path+os.sep, "")
    tbl.loc[idx,"starting frame"] = frame_beg+ifr
    tbl.loc[idx,"ending frame"] = frame_end+ifr
    tbl.loc[idx,"starting time"] = start_time+beg
    tbl.loc[idx,"ending time"] = stop_time+beg
    tbl.loc[idx,"length"] = audiolength
    ## Strip frames from video to specific folder
    imagenames, frame_count = video2frames(
        video_path,
        filename,
        frame_beg+ifr,
        frame_end+ifr,
        sfr,
        frame_path
    )
    tbl.loc[idx,"frames"] = frame_count
    print("{} images extracted from {}".format(frame_count, filename))
    print("from frame {} until frame {}".format(frame_beg+ifr, frame_end+ifr))
    ## Convert image frames to cartoons and add them to GIFs
    images = []
    for names in imagenames:
        image_BGR = cv2.imread(names)[..., :3]
        image_RGB = image_BGR[..., ::-1]
        image_out = infer.inference(image_RGB.copy())
        names = names.replace(frame_path, toons_path)
        names = names.replace(".jpg", ".png")
        cv2.imwrite(names, image_out[..., ::-1])
        image_uint8 = cv2.normalize(image_out, None, 255, 0, 
                                    cv2.NORM_MINMAX, cv2.CV_8U)
        images.append(image_uint8)
    filename = filename.replace(video_path, sprit_path)
    videoclip = moviepy.editor.ImageSequenceClip(images, fps=sfr)
    videoclip.write_videofile(filename, fps=sfr)
    
    videoclip = moviepy.video.io.VideoFileClip.VideoFileClip(filename)
    audioclip = moviepy.audio.io.AudioFileClip.AudioFileClip(audioname)

    #new_audioclip = moviepy.CompositeAudioClip([videoclip.audio, audioclip])
    videoclip.audio = audioclip
    videoclip.write_videofile(filename)
    idx = idx + 1

## Save processing records to CSV
tbl.to_csv(output_csv, index=False)