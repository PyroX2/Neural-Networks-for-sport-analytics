import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    image_rows, image_cols, _ = rgb_image.shape
    landmarks = []
    for i, pose_landmark in enumerate(pose_landmarks_list):
        for landmark in pose_landmark:
            normalized_landmark = landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z)
            landmarks.append([normalized_landmark.x*image_cols,
                             normalized_landmark.y*image_rows,
                             normalized_landmark.z*image_cols])

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmark
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image, np.array(landmarks)


def process_mediapipe(runtype, path, progress_bar):
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
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(length):
                frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detection_result = landmarker.detect(frame)
                pose_landmarker_result, landmarks = draw_landmarks_on_image(
                    frame.numpy_view(), detection_result)
                frames.append(pose_landmarker_result)
                progress_bar.setValue(int(100*i/length))
                ret, frame = video.read()
                # print("Video Processed")
            return frames, None
        elif runtype == 'Image':
            input = mp.Image.create_from_file(path)
            detection_result = landmarker.detect(input)
            # print(detection_result.pose_landmarks)

            annotated_image, landmarks = draw_landmarks_on_image(
                input.numpy_view(), detection_result)
            # return annotated_image
            return annotated_image, landmarks

            # cv2.imshow('a', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # print("Image Processed")


def main():
    output = process_mediapipe(
        'Image', '/home/jakub/inzynierka/app/test_images/image.jpg')


if __name__ == '__main__':
    main()
