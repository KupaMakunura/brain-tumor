import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D, Dropout, Activation, Flatten, Dense


# image directory
image_dir = 'datasets/'

# no tumor list

dataset = []
labels = []
INPUT_SIZE = 64

for i, image_name in enumerate(no_tumor):
    if image_name.split('.')[1] == 'jpg':
        # read the images using open cv
        image = cv2.imread(image_dir + 'no/' + image_name)

        # convert it to rgb using pillow
        image = Image.fromarray(image, 'RGB')

        # resize the image
        image = image.resize((64, 64))
        # set the datasets as numpy arrays

        dataset.append(np.array(image))

        labels.append(0)

# yes tumor list
yes_tumor = os.listdir(image_dir + 'yes/')

for i, image_name in enumerate(yes_tumor):
    if image_name.split('.')[1] == 'jpg':
        # read the images using open cv
        image = cv2.imread(image_dir + 'yes/' + image_name)

        # convert it to rgb using pillow
        image = Image.fromarray(image, 'RGB')

        # resize the image
        image = image.resize((64, 64))
        # set the datasets as numpy arrays

        dataset.append(np.array(image))

        labels.append(1)

# convert the datasets and labels into a numpy array

dataset = np.array(dataset)
labels = np.array(labels)

# 20% data for testing
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2)

print(x_train.shape)
