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

import os
import cv2

def video2frames(video_path, videoname, frame_beg, frame_end, frame_skip, image_path):
    imagename = videoname.replace(video_path, image_path)
    cap = cv2.VideoCapture(videoname)
    ret, frame = cap.read()
    # a variable to keep track of the frame to be saved
    i = 1
    # a variable to keep track of total frames in video
    frame_id = 0
    # a variable to keep track of the frame to be saved
    frame_count = 1
    ## A list for image filenames
    imagenames = []
    extesion = ".mp4"
    while cap.isOpened():
        frame_id += 1
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id == frame_beg:
            filename = imagename.replace(extesion, "_")+str(1)+".jpg"
            imagenames.append(filename)
            cv2.imwrite(filename, frame)
        if frame_id > frame_beg and frame_id <= frame_end:
            if i > frame_skip - 1:
                frame_count += 1
                filename = imagename.replace(extesion, "_")+str(frame_count)+".jpg"
                imagenames.append(filename)
                cv2.imwrite(filename, frame)
                i = 1
                continue
            i += 1
    cap.release()
    return(imagenames,frame_count)