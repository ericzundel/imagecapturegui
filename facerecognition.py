#! env python
"""GUI that loads the face recognition model and does prediction when button is pressed

See model creation code at
https://colab.research.google.com/drive/1AdO1kHuEQfOWgx-fv5d9CbPYj2RmnpMU#scrollTo=2122e422

See example for using tflite on raspberry pi at:
https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi
"""

import os
import platform
import time
import json
import traceback

import numpy as np
import cv2 as cv

# import matplotlib.pyplot as plt
import tkinter as tk
from gtts import gTTS

from ui.main_controller import MainController

# This weird import code is so we can support both the full
# Tensorflow library (linux, windows) and Tensorflow Lite
# (linux).

# Besides the fact that it is good for testing to try both versions,
# Currently, Google doesn't make tensorflow Lite
# binaries available for Windows.

tensorflow_type = None
model = None
interpreter = None
vision = None
"""
try:
    import tensorflow as tf

    tensorflow_type = "FULL"
    print("Loaded Tensorflow Full Version")
except ModuleNotFoundError:
    try:
        import tflite_runtime.interpreter as tflite_runtime

        tensorflow_type = "LITE"
        print("Loaded Tensorflow Lite")
    except ModuleNotFoundError:
        print("Cannot load either tensorflow or tflite_runtime modules")
        exit(1)
"""
###################################################################
# Constants

FACE_RECOGNITION_IMAGE_WIDTH = 100
FACE_RECOGNITION_IMAGE_HEIGHT = 100

# Local path to find the model files
MODEL_PATHNAME_BASE = "./2024model/"
MODEL_FILENAME_BASE = "student_recognition_2024_32bit"
LABEL_FILENAME = "student_recognition_labels.json"

TEST_IMAGE1 = os.path.join(MODEL_PATHNAME_BASE, "donald_test.png")
TEST_IMAGE2 = os.path.join(MODEL_PATHNAME_BASE, "laila_test.png")

DEFAULT_FONT = ("Any", 16)
LIST_HEIGHT = 12  # Number of rows in listbox element
LIST_WIDTH = 20  # Characters wide for listbox element

DISPLAY_IMAGE_WIDTH = 120  # Size of image when displayed on screen
DISPLAY_IMAGE_HEIGHT = 120

DISPLAY_TIMEOUT_SECS = 5


####################################################################
# Setup the Ultrasonic Sensor as a proximity sensor
# (Requires extra hardware - only works on Raspberry Pi)
#
# try:
#    from proximity_sensor import ProximitySensor
#    proximity_sensor = ProximitySensor(echo_pin=17, trigger_pin=4, debug=True)
#    print("Proximity Sensor initialized")
# except BaseException:
#    # It's OK, probably not running on Raspberry Pi
#    proximity_sensor = None
#    print("No proximity sensor detected")

# Our sensor was unreliable, so short cicruit that code.
proximity_sensor = None

try:
    import RPi.GPIO as GPIO
except ModuleNotFoundError:
    print(
        "Error importing RPi.GPIO!  ",
        "This is probably because you are not running on a Raspberry Pi." "Ignoring.",
    )
    GPIO = None
except RuntimeError as e:
    print(
        "Error importing RPi.GPIO!  ",
        "This is probably because you need superuser privileges.  ",
        "You can achieve this by using 'sudo' to run your script",
    )
    raise e
##########################
# Setup the Ultrasonic sensor pins
if GPIO is not None:
    GPIO.setmode(GPIO.BCM)

    push_button_pin = 21

    # Setup IO pin for button
    GPIO.setup(push_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)


############################################################################
# ML code


def load_model():
    global model
    global interpreter

    if tensorflow_type == "FULL":
        print("Initializing Tensorflow Version" + tf.__version__)
        model_path = os.path.join(
            MODEL_PATHNAME_BASE, "%s.tf" % (MODEL_FILENAME_BASE))
        try:
            model = tf.keras.models.load_model(model_path)
        except IOError as ex:
            print("Couldn't find %s. Try unzipping it?" % (model_path))
            exit(1)
        # Sanity check the model after loading
        model.summary()
    elif tensorflow_type == "LITE":
        print("Initializing Tensorflow Lite")
        # Load the TFLite model
        interpreter = tflite_runtime.Interpreter(
            model_path=os.path.join(
                MODEL_PATHNAME_BASE, "%s.tflite" % (MODEL_FILENAME_BASE)
            )
        )
        interpreter.allocate_tensors()


def tensor_from_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Make sure we get back to triplets for RGB to match our model
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # Tensorflow lite requires RGB colorspace
    # img = cv# .cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (FACE_RECOGNITION_IMAGE_WIDTH,
                    FACE_RECOGNITION_IMAGE_HEIGHT))
    arr = my_img_to_arr(img) / 255.0

    # print("Shape of array is: ")
    # print(arr.shape)
    # plt.imshow(arr)
    # plt.show()

    return arr


def predict_lite(interpreter, tensor):

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_tensors = np.array(tensor)
    print("Predict: Test tensor shape")
    print(test_tensors.shape)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]["index"], np.array([test_tensors]))

    # Run the inference
    interpreter.invoke()

    # Extract the output
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print(output_data)
    return output_data[0]


def predict_full(model, tensor):
    test_tensors = np.array(tensor)
    print("Predict: Test tensor shape")
    print(test_tensors.shape)
    prediction = model(np.array([tensor]))
    return prediction[0].numpy()  # Convert from tensor to numpy


def predict(model, interpreter, tensor):
    if tensorflow_type == "FULL":
        return predict_full(model, tensor)
    elif tensorflow_type == "LITE":
        return predict_lite(interpreter, tensor)


def pretty_print_predictions(prediction, labels):
    prediction_arr = prediction
    for i, prob in enumerate(prediction_arr):
        print("%37s: %2.2f%%" % (labels[i], prob * 100.0))


def my_img_to_arr(image):
    # return np.expand_dims(image, axis=0)
    if tensorflow_type == "FULL":
        return tf.keras.utils.img_to_array(image)
    elif tensorflow_type == "LITE":
        return np.asarray(image, dtype=np.float32)

###########################################################
# GUI Code
# See the 'ui' package


###########################################################
# Other functions


def check_button():
    """There is a button attached to GPIO21. See if it is low"""
    if GPIO is not None:
        if GPIO.input(push_button_pin) == GPIO.LOW:
            return True
    return False


def capture_image():
    """Captures a single image from the camera

    Returns: image buffer
    """

    status, frame = camera.read()
    # Throw away the previous frame, it might be cached
    status, frame = camera.read()

    if not status:
        print("Frame is not been captured. Exiting...")
        raise Exception("Frame not captured. Is camera connected?")
    return frame


def load_labels():
    """Load the labels from the json file that correspond to the model outputs"""
    labels_filename = os.path.join(MODEL_PATHNAME_BASE, LABEL_FILENAME)
    with open(labels_filename) as f:
        return json.load(f)


def text_to_speech(text):
    tts = gTTS(text=text, lang="en")
    filename = "speech.mp3"
    tts.save(filename)
    platform_name = platform.system()
    if platform_name == "Windows":
        os.system(f"start {filename}")
    elif platform_name == "Linux":
        os.system(f"mplayer {filename}")
    else:
        print("Update script for how to play sound on %s" % platform_name)


def do_predict(img, labels):
    tensor = tensor_from_image(img)

    print("tensor is:")
    print(tensor)

    # Run the image through the model to see which output it predicts
    prediction = predict(model, interpreter, tensor)

    # import pdb; pdb.set_trace()
    pretty_print_predictions(prediction, labels)

    # Display the entry with the highest probability
    highest_prediction_index = prediction.argmax()
    certainty = float(prediction[highest_prediction_index])
    print(
        "Prediction %d %s  Certainty: %0.2f"
        % (
            highest_prediction_index,
            labels[highest_prediction_index],
            certainty,
        )
    )

    # Say the name out loud
    first_name = labels[highest_prediction_index].split(sep="_")[0]
    text_to_speech("Hello, %s" % (first_name))

    return labels[highest_prediction_index], certainty


def test_and_predict(image_filename, labels):
    img = cv.imread(image_filename, cv.IMREAD_COLOR)
    return do_predict(img, labels)


def capture_and_predict(labels):
    """Grab an image and run it through the ML model

    Returns: predicted_name, certainty  where certainty is a value between 0 and 1.0
    """
    # Grab an image from the camera and transform it to a tensor to feed into the model
    img = capture_image()
    return do_predict(img, labels)


#####
#
labels = load_labels()

####################################################################
# Setup OpenCV for reading from the camera
#
camera = cv.VideoCapture(0)

# Important: Turn off the buffer on the camera. Otherwise, you get stale images
camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

# Create and display the main UI
controller = MainController(labels=labels)
controller.set_state("WAITING")


#####################################################################
# Initialize the Machine Learning model. This takes some time (about 20 seconds)
print("*** TODO: load the model")
# load_model()

try:
    controller.mainloop()

except BaseException as e:
    print("Exiting due to %s " % str(e))
    print(traceback.format_exc())

# When everything done, release resources.
controller.close_window()

camera.release()
if proximity_sensor is not None:
    proximity_sensor.deinit()
