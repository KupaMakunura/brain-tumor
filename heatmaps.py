import cv2

from ultralytics import solutions


def generate_heatmap_explanation(img_path):

    img = cv2.imread(img_path)

    # initialize the heatmap
    heatmap = solutions.Heatmap(
        show=True,
        model="tumor-yolo11vn.pt",
        colormap=cv2.COLORMAP_PARULA,
    )

    im0 = heatmap.generate_heatmap(img)

    # write the output to the file

    cv2.imwrite("heatmap_output.jpg", im0)
