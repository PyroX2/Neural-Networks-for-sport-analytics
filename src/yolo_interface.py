from ultralytics import YOLO
import cv2
import torch
import src.utils as utils

# Define skeleton for drawing the whole pose
SKELETON = ((0, 1),
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
            (12, 11))


def process_yolo(runtype: str, path: str, progress_bar_function) -> tuple[list, list[torch.Tensor], str]:
    # Load the model
    model = YOLO('models/yolo/yolov8n-pose.pt')

    # Process depending on runtype
    if runtype == 'Video':
        # Process video by neural network
        results = model(path, stream=True)

        # Capture video with cv2
        video = cv2.VideoCapture(path)

        # Get total number of frames
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Lists for storing output images and landmarks
        output_images = []
        landmarks = []  # List of torch tensors

        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            # Draw keypoints
            output_image = utils.draw_keypoints(
                keypoints.xy[0], result.orig_img, SKELETON)

            # Change BGR to RGB
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            # Append frame to list
            output_images.append(output_image)

            # Use hstack to add zeros for Z dim
            landmarks.append(torch.hstack(
                (keypoints.xy[0], torch.tensor([[0]]*len(keypoints.xy[0])))))

            progress_bar_function('YOLO', int(i/length*100))
        return output_images, landmarks, 'YOLO'

    elif runtype == 'Image':
        # Process the image
        results = model(path)

        for result in results:
            # Extract keypoints
            keypoints = result.keypoints

            # Draw keypoints on image
            output_image = utils.draw_keypoints(
                keypoints.xy[0], result.orig_img, SKELETON)

            # Change BGR to RGB
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            '''
            Use hstack to add zeros for Z dim
            List with torch tensor so that the type matches the type of video output
            '''
            landmarks = [(torch.hstack(
                (keypoints.xy[0], torch.tensor([[0]]*len(keypoints.xy[0])))))]
        return [output_image], landmarks, 'YOLO'


# Main just to quickly test this module functionality
def main():
    process_yolo('Image', '/home/jakub/inzynierka/app/test_images/image.jpg')


if __name__ == '__main__':
    main()
