from ultralytics import YOLO


def process_yolo(runtype, path, progress_bar):
    model = YOLO('models/yolo/yolov8n-pose.pt')

    results = model(path, stream=True)
    landmarks = []
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        landmarks.append([keypoints.xy[0], keypoints.xy[1], 0])
    return results, landmarks
    # result.show()  # display to screen
    # result.save(filename='result.jpg')  # save to disk
