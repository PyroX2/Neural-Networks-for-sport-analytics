#! /bin/bash

mkdir -p models/mediapipe
mkdir  test_images
wget -O models/mediapipe/pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
wget -q -O test_images/image.jpg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg
