import dis
from ultralytics import YOLO, solutions
import cv2


class ModelPredictor:
    def __init__(self, image_path):
        self.model = YOLO("tumor-yolov11n.pt")
        self.confidences = []
        self.boxes = []
        self.class_labels = []
        self.image_path = image_path

    def predict(
        self,
    ):
        # set the model params
        return self.model.predict(
            self.image_path,
            conf=0.25,
            classes=[1],
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

        cv2.imwrite("heatmap_output.jpg", im0)


model = ModelPredictor("datasets/yes/y200.jpg")


results = model.predict()

model.set_params(results)

# generate the heatmap
model.generate_heatmap_explanation()
