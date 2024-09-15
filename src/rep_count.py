import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import src.utils as utils
from src.limbs_keypoints import LIMBS_KEYPOINTS
from src.skeletons import SKELTONS


class RepCount:
    def __init__(self):
        self.lower_bound = 50
        self.upper_bound = 160
        self.rep_count = 0
        self.state = 0

    def calculate_angle(self, keypoints: torch.Tensor):
        p1 = keypoints[0]
        p2 = keypoints[1]
        p3 = keypoints[2]

        p1 = p1 - p2
        p3 = p3 - p2

        alfa = np.rad2deg(np.arctan2(p1[1], p1[0]))
        beta = np.rad2deg(np.arctan2(p3[1], p3[0]))

        alfa += 360
        if beta < 0:
            beta += 360

        correct = self.is_correct(alfa, beta)
        if not correct:
            self.state = 2
        self.update_count(alfa, beta)
        return alfa, beta, correct

    def is_correct(self, alfa, beta):
        if np.abs(beta - alfa) < 40:
            return False
        elif np.abs(beta - alfa) > 180:
            return False
        else:
            return True

    def update_count(self, alfa, beta):
        gamma = np.abs(beta - alfa)
        if self.state == 0 and gamma < self.lower_bound:
            self.state = 1
        elif self.state == 1 and gamma > self.upper_bound:
            self.state = 0
            self.rep_count += 1
        elif self.state == 2 and gamma > self.upper_bound:
            self.state = 0

    def process(self, frame: np.ndarray, keypoints: torch.Tensor, limb: str):
        color = (0, 255, 0)

        if len(keypoints) == 0:
            return frame

        selected_keypoints, skeleton = utils.extract_given_keypoints(
            keypoints, SKELTONS['YOLO'], LIMBS_KEYPOINTS[limb])

        for keypoint in selected_keypoints:
            if not (keypoint).all():
                return frame

        alfa, beta, correct = self.calculate_angle(selected_keypoints)

        center = (int(selected_keypoints[1][0]), int(selected_keypoints[1][1]))
        text = 'Correct'
        if not correct:
            color = (0, 0, 255)
            text = 'Incorrect'
        frame = cv2.ellipse(
            img=frame, center=center, axes=(50, 50), angle=0, startAngle=int(alfa), endAngle=int(beta), color=color, thickness=5)
        frame = cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color)

        return frame
