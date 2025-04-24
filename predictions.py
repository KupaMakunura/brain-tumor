import uuid

import cv2
import numpy as np
import requests
import supervision as sv


class ModelPredictor:
    def __init__(self, image_path):
        self.api_url = "https://predict.ultralytics.com"
        self.api_key = "fba500f7b229869c21a3551518bde7fa9a0fc270f9"
        self.image_path = image_path
        self.detections = []
        self.results = None
        self.result_image_path = None

    def predict(self):
        headers = {"x-api-key": self.api_key}
        data = {
            "model": "https://hub.ultralytics.com/models/9KfmbQpWm588VVkYP03h",
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
        }

        with open(self.image_path, "rb") as f:
            response = requests.post(
                self.api_url, headers=headers, data=data, files={"file": f}
            )

        response.raise_for_status()
        self.results = response.json()

    def set_params(self):
        if not self.results:
            return

        for image in self.results.get("images", []):
            for detection in image.get("results", []):

                if detection.get("name") == "positive":
                    box = detection["box"]
                    width = box["x2"] - box["x1"]
                    height = box["y2"] - box["y1"]
                    tumor_size = width * height

                    self.detections.append(
                        {
                            "bounding_box": {
                                "x_min": box["x1"],
                                "y_min": box["y1"],
                                "x_max": box["x2"],
                                "y_max": box["y2"],
                            },
                            "confidence": detection["confidence"],
                            "class_label": "tumor",
                            "tumor_size": tumor_size,
                        }
                    )

    def plot_image(self):
        if not self.results or not self.detections:
            self.result_image_path = None
            return

        # Read image
        img = cv2.imread(self.image_path)

        # Convert detections to supervision format
        boxes = []
        confidences = []
        class_ids = []

        for detection in self.detections:
            box = detection["bounding_box"]
            boxes.append([box["x_min"], box["y_min"], box["x_max"], box["y_max"]])
            confidences.append(detection["confidence"])
            class_ids.append(0)  # 0 for tumor class

        # Create supervision detections
        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
        )

        # Create box annotator
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator()

        # Draw detections
        frame = box_annotator.annotate(
            scene=img,
            detections=detections,
        )
        # label the frame
        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=[f"tumor {confidence:.2f}" for confidence in confidences],
        )

        self.result_image_path = f"uploads/predictions/{uuid.uuid4()}.jpg"

        # Save annotated image
        cv2.imwrite(self.result_image_path, frame)

        # Save result image path
