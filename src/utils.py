import cv2
import random


def draw_keypoints(landmarks, image, skeleton):
    for start_point, end_point in skeleton:
        if landmarks[start_point].all() == 0 or landmarks[end_point].all() == 0:
            continue
        image = cv2.line(
            image, (int(landmarks[start_point][0].item()),
                    int(landmarks[start_point][1].item())),
            (int(landmarks[end_point][0].item()),
             int(landmarks[end_point][1].item())),
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 4)

    return image


def draw_points(landmarks, image):
    for i, landmark in enumerate(landmarks):
        image = cv2.circle(
            image, (int(landmark[0].item()), int(landmark[1].item())), 2, (255, 0, 0), 2)
        image = cv2.putText(image, str(
            i), (int(landmark[0].item()), int(landmark[1].item())), cv2.FONT_HERSHEY_COMPLEX, 1, color=(255, 0, 0))
    return image
