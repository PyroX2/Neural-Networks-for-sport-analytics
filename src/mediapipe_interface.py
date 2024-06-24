import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2


def draw_landmarks_on_image(rgb_image: np.array,
                            detection_result: mp.tasks.vision.PoseLandmarkerResult) -> tuple[np.array, list]:
    # Extract list with landmarks and create a copy of an image
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    image_rows, image_cols, _ = rgb_image.shape

    # List for all landmarks with size (number_of_detections, number_of_landmarks, 3)
    landmarks = []

    # Get every person's landmarks
    for i, pose_landmark in enumerate(pose_landmarks_list):
        for landmark in pose_landmark:  # Get every keypoint
            # Extract landmarks
            normalized_landmark = landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z)

            # For each landmark crate a list with XYZ coordinates of a point
            landmarks.append([normalized_landmark.x*image_cols,
                             normalized_landmark.y*image_rows,
                             normalized_landmark.z*image_cols])

        # Draw the pose landmarks using mediapipe functions
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmark
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())

    return annotated_image, landmarks


# Function for processing of images and videos
def process_mediapipe(runtype: str, path: str) -> tuple[list, list, str]:
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Set base model
    base_options = python.BaseOptions(
        model_asset_path='./models/mediapipe/pose_landmarker.task')

    # Set options for processing images and videos
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=VisionRunningMode.IMAGE)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # Processing pipeline for videos
        if runtype == 'Video':

            # List to store all landmarks and frames
            all_landmarks = []
            frames = []

            # Read video
            video = cv2.VideoCapture(path)

            # Read video's FPS
            fps = video.get(cv2.CAP_PROP_FPS)

            # Read first video frame
            ret, frame = video.read()

            # Read video length
            length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # Loop for processing each frame
            for i in range(length):
                # Read image as mp.Image type
                frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                # Process single image
                detection_result = landmarker.detect(
                    frame)

                # Read image as numpy array
                numpy_frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Draw landmarks on an image
                pose_landmarker_result, landmarks = draw_landmarks_on_image(
                    numpy_frame, detection_result)

                # Append results to lists
                frames.append(pose_landmarker_result)
                all_landmarks.append(landmarks)

                # Read next frame
                ret, frame = video.read()
            return frames, all_landmarks, 'MediaPipe'

        # Processing pipeline for images
        elif runtype == 'Image':
            input = mp.Image.create_from_file(path)
            detection_result = landmarker.detect(input)
            annotated_image, landmarks = draw_landmarks_on_image(
                input.numpy_view(), detection_result)
            return [annotated_image], landmarks, 'MediaPipe'


# Just to test this module on its own
def main():
    output = process_mediapipe(
        'Image', '/home/jakub/inzynierka/app/test_images/image.jpg')


if __name__ == '__main__':
    main()
