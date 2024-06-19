from ultralytics import YOLO
import cv2
import torch
import utils

SKELETON = [(0, 1),
            (1, 3),
            (0, 2),
            (2, 4),
            (3, 5),
            (4, 6),
            (6, 8),
            (8, 10),
            (5, 7),
            (7, 9),
            (6, 12),
            (5, 11),
            (12, 14),
            (14, 16),
            (11, 13),
            (13, 15),
            (12, 11)]


def process_yolo(runtype, path):
    model = YOLO('models/yolo/yolov8n-pose.pt')

    if runtype == 'Video':
        results = model(path, stream=True)
        video = cv2.VideoCapture(path)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        output_images = []
        landmarks = []
        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            output_image = utils.draw_keypoints(
                keypoints.xy[0], result.orig_img, SKELETON)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            output_images.append(output_image)
            landmarks.append(torch.hstack(
                (keypoints.xy[0], torch.tensor([[0]]*len(keypoints.xy[0])))))
        return output_images, landmarks, 'YOLO'
    elif runtype == 'Image':
        results = model(path)
        for result in results:
            keypoints = result.keypoints
            output_image = utils.draw_keypoints(
                keypoints.xy[0], result.orig_img, SKELETON)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            landmarks = [(torch.hstack(
                (keypoints.xy[0], torch.tensor([[0]]*len(keypoints.xy[0])))))]
        return [output_image], landmarks, 'YOLO'


def main():
    process_yolo('Image', '/home/jakub/inzynierka/app/test_images/image.jpg')


if __name__ == '__main__':
    main()
