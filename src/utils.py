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


def draw_keypoints(landmarks: torch.Tensor, image: np.ndarray, skeleton: tuple) -> np.ndarray:
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


def scale_frame(frame: np.ndarray, x_max: int, y_max: int) -> np.ndarray:
    # Read X and Y axis sizes
    x_size, y_size = frame.shape[0], frame.shape[1]

    if x_size > x_max:
        y_size = y_size * (x_max / x_size)  # y_size times scaling factor
        x_size = x_max

    # This is not elif because y_size might be still to big after x axis scaling
    if y_size > y_max:
        x_size = x_size * (y_max / y_size)  # x_size times scaling factor
        y_size = y_max

    # Resize frame
    resized_frame = cv2.resize(frame, dsize=(
        int(y_size), int(x_size)), interpolation=cv2.INTER_CUBIC)
    # resized_frame = resize(frame, (x_size, y_size))
    return resized_frame


def keypoints_eq_0(start_point: torch.Tensor, end_point: torch.Tensor) -> bool:
    zeros_tensor = torch.tensor([0, 0, 0])
    return torch.eq(start_point, zeros_tensor).all() == True or torch.eq(end_point, zeros_tensor).all() == True


def calculate_least_squares(y: list[float], x: list[float], n: int) -> list:
    x = np.array(x)

    H = np.empty((len(y), n + 1))

    for i in range(n + 1):
        H[:, i] = x ** i

    y = np.array(y)[np.newaxis]
    H_inversed = np.linalg.inv(H.transpose().dot(H))
    params = H_inversed.dot(H.transpose()).dot(y.T)

    return params


def calculate_new_vector(current_value: torch.Tensor, prev_keypoint_position: torch.Tensor, fps: float) -> list[float, float]:
    new_vector_x = (current_value[0] - prev_keypoint_position[0]) * fps
    new_vector_y = (current_value[1] - prev_keypoint_position[1]) * fps
    return [new_vector_x, new_vector_y]


def extract_given_keypoints(keypoints: torch.Tensor, skeleton: list, indicies: list) -> tuple:
    selected_keypoints = keypoints[indicies]
    selected_keypoints = selected_keypoints[:, :2]
    selected_skeleton = []

    for link in skeleton:
        if link[0] in indicies and link[1] in indicies:
            selected_skeleton.append(link)

    return selected_keypoints, selected_skeleton
