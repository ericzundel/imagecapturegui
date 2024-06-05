"""Simple code to test loading the face recognition model and evaluating an image

See model creation code at  https://colab.research.google.com/drive/1AdO1kHuEQfOWgx-fv5d9CbPYj2RmnpMU#scrollTo=2122e422
"""

import os
import platform

# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.datasets import imdb
# import numpy as np
# import os, shutil, pathlib
# from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

# from tensorflow.keras.utils import array_to_img
# import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import cv2 as cv

# import PySimpleGUI as sg

from gtts import gTTS

import json

FACE_RECOGNITION_IMAGE_WIDTH = 100
FACE_RECOGNITION_IMAGE_HEIGHT = 100
MODEL_PATHNAME = "./2024model/"

student_labels = []


def load_labels():
    global student_labels
    labels_filename = os.path.join(
        MODEL_PATHNAME, "student_recognition_labels.json")
    labels_file = open(labels_filename)

    json_data = json.load(labels_file)
    student_labels = json_data
    print("Student Labels:")
    print(student_labels)


def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    filename = "speech.mp3"
    tts.save(filename)
    platform_name = platform.system()
    if platform_name == "Windows":
        os.system(f"start {filename}")
    elif platform_name == "Linux":
        os.system(f"aplay {filename}")
    else:
        print("Update script for how to play sound on %s" % platform_name)


def tensor_from_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rgb_gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    # small_image = cv.resize(rgb_gray,
    #                   (FACE_RECOGNITION_IMAGE_WIDTH, FACE_RECOGNITION_IMAGE_HEIGHT))
    arr = img_to_array(rgb_gray)
    arr = np.resize(
        arr, (FACE_RECOGNITION_IMAGE_WIDTH, FACE_RECOGNITION_IMAGE_HEIGHT, 3)
    )
    print("Shape of array is: ")
    print(arr.shape)
    new_arr = arr.reshape(
        (FACE_RECOGNITION_IMAGE_WIDTH, FACE_RECOGNITION_IMAGE_HEIGHT, 3)
    )

    print("Shape of array is: ")
    print(arr.shape)
    return np.float32(new_arr)


def capture_image():
    """Captures a single image from the camera

    Returns: image buffer
    """

    status, frame = camera.read()

    if not status:
        print("Frame is not been captured. Exiting...")
        raise Exception("Frame not captured. Is camera connected?")
    return frame


def predict(model, tensor):
    test_tensors = np.array(tensor)
    print("Predict: Test tensor shape")
    print(test_tensors.shape)
    prediction = model(np.array([tensor]))
    return prediction[0]


def pretty_print_predictions(prediction):
    prediction_arr = prediction.numpy()
    for i, prob in enumerate(prediction_arr):
        print("%37s: %2.2f%%" % (student_labels[i], prob * 100.0))


#####
#
load_labels()

####################################################################
# Setup OpenCV for reading from the camera
#
camera = cv.VideoCapture(0)

# Important: Turn off the buffer on the camera. Otherwise, you get stale images
camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

print("Tensorflow Version" + tf.__version__)

model = tf.keras.models.load_model(
    os.path.join(MODEL_PATHNAME, "student_recognition_2024_32bit.tf")
)

# Sanity check the model after loading
model.summary()

# Grab an image from the camera and transform it to a tensor to feed into the model
img = capture_image()
tensor = tensor_from_image(img)
# Normalize to keras image format which use datapoints with float values from 0-1.0
tensor = tensor / 255.0

# Run the image through the model to see which output it predicts
prediction = predict(model, tensor)

pretty_print_predictions(prediction)

# Display the entry with the highest probability
highest_prediction_index = prediction.numpy().argmax()
certainty = prediction[highest_prediction_index]
print(
    "Prediction %d %s  Probability: %0.2f"
    % (highest_prediction_index, student_labels[highest_prediction_index], certainty)
)
