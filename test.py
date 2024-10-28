import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

INPUT_SIZE = 64

model = load_model('braintumordetector10ECategorical.h5')

image = cv2.imread('./datasets/yes/y0.jpg')

img = Image.fromarray(image)

img = img.resize((INPUT_SIZE, INPUT_SIZE))

img = np.array(img)


result = model.predict_step(img)

print(result)