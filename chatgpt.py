import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

ZONE_POLYGON = np.array([
    [0, 0],
    [640 // 2, 0],
    [640 // 2, 480 // 2],
    [0, 480]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    print("hello")
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone = sv.PolygonZone(polygon=ZONE_POLYGON, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.red())

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Split the frame into two ROIs
        frame_left = frame[:, :frame_width // 2]
        frame_right = frame[:, frame_width // 2:]

        # Apply object detection and annotation to the left ROI
        result_left = model(frame_left)[0]
        detections_left = sv.Detections.from_yolov8(result_left)
        labels_left = [f"{model.names[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_left]
        frame_left = box_annotator.annotate(scene=frame_left, detections=detections_left, labels=labels_left)
        zone.trigger(detections=detections_left)
        frame_left = zone_annotator.annotate(scene=frame_left)

        # Apply object detection and annotation to the right ROI
        result_right = model(frame_right)[0]
        detections_right = sv.Detections.from_yolov8(result_right)
        labels_right = [f"{model.names[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections_right]
        frame_right = box_annotator.annotate(scene=frame_right, detections=detections_right, labels=labels_right)

        # Combine the two ROIs back into the full frame
        frame_combined = np.hstack((frame_left, frame_right))

        cv2.imshow("yolov8", frame_combined)

        if cv2.waitKey(30) == 27:
            break

if __name__ == "__main__":
    main()
