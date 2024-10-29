from matplotlib.pyplot import cla
import numpy as np
from ultralytics import YOLO


class ModelPredictor:
    def __init__(self):
        self.model = YOLO("tumor-yolov11n.pt")
        self.confidences = []
        self.boxes = []
        self.class_labels = []

    def predict(self, image_path):
        # set the model params
        return self.model.predict(
            image_path,
            conf=0.25,
            classes=[0, 1],
            iou=0.45,
            retina_masks=True,
        )

    # set params to the class variables

    def set_params(self, results):

        for result in results:

            boxes = result.boxes.cpu().numpy()

            self.boxes.append(boxes.xyxy)
            self.confidences.append(boxes.conf)
            self.class_labels.append(boxes.cls)

            # call the plot method
            self.plot_image("result.jpg", result)

    # draw the boxes on the image

    def plot_image(self, image_path, result):
        return result.save(filename=image_path)

    # return the labels ,boxes and confidences

    def get_params(self):
        return self.boxes, self.confidences, self.class_labels


model = ModelPredictor()


results = model.predict("datasets/yes/y210.jpg")

model.set_params(results)


print(model.get_params())
