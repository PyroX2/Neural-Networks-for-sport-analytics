import cv2
import numpy as np
import torch
from skimage.transform import resize


# Define colors for each connection in skeleton
SKELETON_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 255, 128),
    (255, 0, 128),
    (128, 255, 0),
    (0, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 255),
    (255, 255, 128),
    (255, 128, 255),
    (128, 255, 255),
    (255, 128, 64),
    (128, 64, 255)
)


def draw_keypoints(landmarks: torch.Tensor, image: np.array, skeleton: tuple) -> np.array:
    # If landmarks list is empty return original image
    if len(landmarks) == 0:
        return image

    # Extract every point
    for i, (start_point, end_point) in enumerate(skeleton):
        # Skip point if it is 0
        if landmarks[start_point].all() == 0 or landmarks[end_point].all() == 0:
            continue

        # If landmark exists draw it on image
        image = cv2.line(
            image, (int(landmarks[start_point][0].item()),
                    int(landmarks[start_point][1].item())),
            (int(landmarks[end_point][0].item()),
             int(landmarks[end_point][1].item())),
            SKELETON_COLORS[i], 4)
    return image


def scale_frame(frame: np.array, x_max: int, y_max: int) -> np.array:
    # Read X and Y axis sizes
    x_size, y_size = frame.shape[0], frame.shape[1]

    if x_size > x_max:
        y_size = y_size*(x_max/x_size)  # y_size times scaling factor
        x_size = x_max

    # This is not elif because y_size might be still to big after x axis scaling
    if y_size > y_max:
        x_size = x_size*(y_max/y_size)  # x_size times scaling factor
        y_size = y_max

    # Resize frame
    resized_frame = cv2.resize(frame, dsize=(
        int(y_size), int(x_size)), interpolation=cv2.INTER_CUBIC)
    # resized_frame = resize(frame, (x_size, y_size))
    return resized_frame
