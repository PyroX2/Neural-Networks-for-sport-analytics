from ultralytics import YOLO
import cv2
import torch


def draw_keypoints(landmarks, image):
    for landmark in landmarks:
        image = cv2.circle(
            image, (int(landmark[0].item()), int(landmark[1].item())), 2, (255, 0, 0), 2)
    return image


def process_yolo(runtype, path, progress_bar=None):
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
            output_image = draw_keypoints(keypoints.xy[0], result.orig_img)
            output_images.append(output_image)
            landmarks.append(torch.hstack(
                (keypoints.xy[0], torch.tensor([[0]]*17))))
            progress_bar.setValue(int(100*i/length))
        return output_images, landmarks
    elif runtype == 'Image':
        results = model(path)
        for result in results:
            keypoints = result.keypoints
            output_image = draw_keypoints(keypoints.xy[0], result.orig_img)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            output_images.append(output_image)
            landmarks.append(torch.hstack(
                (keypoints.xy[0], torch.tensor([[0]]*17))))
        return output_images, landmarks


def main():
    process_yolo('Video', '/home/jakub/Videos/training.mp4')


if __name__ == '__main__':
    main()
