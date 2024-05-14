import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def process_mediapipe(runtype, path=None):
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode

    base_options = python.BaseOptions(
        model_asset_path='./models/mediapipe/pose_landmarker.task')

    if runtype == "Video":
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            running_mode=VisionRunningMode.IMAGE)
    elif runtype == "Image":
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            running_mode=VisionRunningMode.IMAGE)

    with PoseLandmarker.create_from_options(options) as landmarker:
        if runtype == 'Video':
            video = cv2.VideoCapture(path)
            fps = video.get(cv2.CAP_PROP_FPS)

            frames = []
            ret, frame = video.read()
            while ret:
                frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = landmarker.detect(frame)
                pose_landmarker_result = draw_landmarks_on_image(
                    frame.numpy_view(), detection_result)
                frames.append(pose_landmarker_result)

                # cv2.imshow('a', pose_landmarker_result)
                # cv2.waitKey(int(1000/fps))
                ret, frame = video.read()
                # print("Video Processed")
            return frames
        elif runtype == 'Image':
            input = mp.Image.create_from_file(path)
            detection_result = landmarker.detect(input)

            annotated_image = draw_landmarks_on_image(
                input.numpy_view(), detection_result)
            return annotated_image

            # cv2.imshow('a', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # print("Image Processed")
