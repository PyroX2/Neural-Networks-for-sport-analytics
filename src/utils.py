import cv2
import numpy as np
import torch

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
