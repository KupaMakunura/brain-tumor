from torch import mode
from ultralytics import YOLO, solutions
import cv2


class ModelPredictor:
    def __init__(self, image_path):
        self.model = YOLO("tumor-yolov11n.pt")
        self.image_path = image_path
        self.detections = []
        self.results = None
        self.result = None

    def predict(self):
        # set the model params
        self.results = self.model.predict(
            self.image_path,
            conf=0.25,
            classes=[1],
            iou=0.45,
            retina_masks=True,
        )

        return self.results

    # set params to the class variables

    def set_params(self):

        for result in self.results:

            self.result = result
            boxes = result.boxes.cpu().numpy()

            # Structure detection information
            for i in range(len(boxes.xyxy)):
                x_min, y_min, x_max, y_max = boxes.xyxy[i].tolist()
                width, height = x_max - x_min, y_max - y_min
                tumor_size = width * height
                confidence = boxes.conf[i].item()
                class_label = "tumor" if boxes.cls[i] == 1 else "normal"

                self.detections.append(
                    {
                        "bounding_box": {
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                        },
                        "confidence": confidence,
                        "class_label": class_label,
                        "tumor_size": tumor_size,
                    }
                )

    # draw the boxes on the image

    def plot_image(self):
        return self.result.save(filename="results/predictions/prediction.jpg")

    # return the labels ,boxes and confidences

    def get_params(self):
        return self.detections

    def generate_heatmap_explanation(self):

        img = cv2.imread(self.image_path)

        # initialize the heatmap
        heatmap = solutions.Heatmap(
            model="tumor-yolov11n.pt",
            classes=[1],
            colormap=cv2.COLORMAP_PARULA,
        )

        im0 = heatmap.generate_heatmap(img)

        # write the output to the file

        cv2.imwrite("results/heatmaps/heatmap.jpg", im0)
