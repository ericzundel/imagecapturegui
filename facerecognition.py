"""GUI that loads the face recognition model and does prediction when button is pressed

See model creation code at
https://colab.research.google.com/drive/1AdO1kHuEQfOWgx-fv5d9CbPYj2RmnpMU#scrollTo=2122e422
"""

import os
import platform
import time
import json

import numpy as np
import cv2 as cv
import PySimpleGUI as sg
from gtts import gTTS

tensorflow_type = None
model = None
interpretor = None

try:
    import tensorflow as tf

    tensorflow_type = "FULL"
except ModuleNotFoundError:
    try:
        import tflite_runtime.interpreter as tflite

        tensorflow_type = "LITE"
    except ModuleNotFoundError:
        print("Cannot load either tensorflow or tflite_runtime modules")
        exit(1)


FACE_RECOGNITION_IMAGE_WIDTH = 50
FACE_RECOGNITION_IMAGE_HEIGHT = 50
MODEL_PATHNAME = "./2024model/"

DEFAULT_FONT = ("Any", 16)
LIST_HEIGHT = 12  # Number of rows in listbox element
LIST_WIDTH = 20  # Characters wide for listbox element

DISPLAY_IMAGE_WIDTH = 120  # Size of image when displayed on screen
DISPLAY_IMAGE_HEIGHT = 120

DISPLAY_TIMEOUT_SECS = 5

student_labels = []


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

# Our sensor was unreliable...
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
    if tensorflow_type == "FULL":
        print("Tensorflow Version" + tf.__version__)

        model = tf.keras.models.load_model(
            os.path.join(MODEL_PATHNAME, "student_recognition.tf")
        )
        # Sanity check the model after loading
        model.summary()
    elif tensorflow_type == "LITE":
        print("Tensorflow Lite")
        # Load the TFLite model
        interpreter = tflite.Interpreter(
            model_path=os.path.join(MODEL_PATHNAME, "student_recognition.tflite")
        )
        interpreter.allocate_tensors()


def tensor_from_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Make sure we get back to triplets for RGB to match our model
    rgb_gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    arr = my_img_to_arr(rgb_gray)
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
    return new_arr


def predict_lite(interpreter, tensor):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]["index"], tensor)

    # Run the inference
    interpreter.invoke()

    # Extract the output
    output_data = interpreter.get_tensor(output_details[0]["index"])
    print(output_data)


def predict_full(model, tensor):
    test_tensors = np.array(tensor)
    print("Predict: Test tensor shape")
    print(test_tensors.shape)
    prediction = model(np.array([tensor]))
    return prediction[0]


def predict(model, tensor):
    if tensorflow_type == "FULL":
        return predict_full(model, tensor)
    elif tensorflow_type == "LITE":
        return predict_lite(interpretor, tensor)


def pretty_print_predictions(prediction):
    prediction_arr = prediction.numpy()
    for i, prob in enumerate(prediction_arr):
        print("%37s: %2.2f%%" % (student_labels[i], prob * 100.0))


def my_img_to_arr(image):
    return np.expand_dims(image, axis=0)


#############################################################################
#  GUI code
def build_window():
    """Builds the user interface and pops it up on the screen.

    Returns: sg.Window object
    """
    left_column = sg.Column(
        [
            [sg.Text(size=(18, 1), key="-STATUS-", font=DEFAULT_FONT)],
            [sg.pin(sg.Button("Manual Capture", key="-CAPTURE-", font=DEFAULT_FONT))],
            [sg.Text()],  # vertical spacer
            [sg.Text()],  # vertical spacer
            [sg.Text()],  # vertical spacer
            [sg.Button("Exit", font=("Any", 6))],
        ],
        key="-LEFT_COLUMN-",
    )
    right_column = sg.Column(
        [
            [
                sg.Text("Name: ", font=DEFAULT_FONT),
                sg.Text(key="-FACE_NAME-", font=DEFAULT_FONT),
            ],
            [
                sg.Text("Certainty: ", font=DEFAULT_FONT),
                sg.Text(key="-CERTAINTY-", font=DEFAULT_FONT),
            ],
            [sg.Button("Cancel", key="-CANCEL-", font=DEFAULT_FONT)],
        ],
        key="-RIGHT_COLUMN-",
        visible=False,
    )
    # Push and VPush elements help UI to center when the window is maximized
    layout = [
        [sg.VPush()],
        [sg.Push(), sg.Column([[left_column, sg.pin(right_column)]]), sg.Push()],
        [sg.VPush()],
    ]
    window = sg.Window("Face Image Capture", layout, finalize=True, resizable=True)
    # Doing this makes the app take up the whole screen
    window.maximize()
    return window


def set_ui_state(window, state, face_name=None, certainty=None):
    """Set the UI into a specified state.

    state: one of ['WAITING', 'CAPTURING', 'NAMING']

    In state 'WAITING', most of the UI is hidden, but there is a
    button for manually capturing an image presented.

    In state 'CAPTURING', the manual button is hidden and the
    status label is updated

    In state 'NAMING', the captured images are displayed and
    a listbox with choices for choosing the names associated
    with the images is displayed.

    Returns: None
    """

    if state == "WAITING":
        # Hide images and right column
        # Show manual capture button
        window["-STATUS-"].update("Waiting to Capture")
        window["-CAPTURE-"].update(visible=True)
        window["-RIGHT_COLUMN-"].update(visible=False)
        window["-LEFT_COLUMN-"].expand(True, True)
    elif state == "CAPTURING":
        # Hide Manual capture button
        window["-STATUS-"].update("Running ML prediction")
        window["-CAPTURE-"].update(visible=False)
    elif state == "NAMING":
        # Turn on the right column
        window["-STATUS-"].update("Displaying Result")
        window["-RIGHT_COLUMN-"].update(visible=True)
        window["-FACE_NAME-"].update(face_name)
        window["-CERTAINTY-"].update("%2f" % (certainty * 100.0))
        window["-CAPTURE-"].update(visible=False)
    else:
        raise RuntimeError("Invalid state %s" % state)


def main_loop():
    """UI Event Loop

    This loop executes until someone closes the main window.
    """

    last_captured_image_time = 0

    while True:
        # Check for a trigger 4x a second
        event, values = window.read(timeout=50)

        # check for a timeout, send the GUI back to WAITING mode
        if (
            last_captured_image_time > 0
            and time.monotonic() - last_captured_image_time > DISPLAY_TIMEOUT_SECS
        ):
            set_ui_state(window, "WAITING")
            last_captured_image_time = 0

        # Every time something happens in the UI, it returns an event.
        # By decoding this event you can figure out what happened and take
        # an action.
        if event in (sg.WIN_CLOSED, "Exit"):  # always check for closed window
            break
        # If the user doesn't want to classify this image, the cancel button
        # will clear out the state.
        if event == "-CANCEL-":
            set_ui_state(window, "WAITING")
        # Check to see if we are to capture new images by checking the
        # proximity sensor hardware or if the button was pressed
        if check_button() or event == "-CAPTURE-":
            set_ui_state(window, "CAPTURING")
            last_captured_image_time = time.monotonic()
            name, certainty = capture_and_predict()
            set_ui_state(window, "NAMING", face_name=name, certainty=certainty)


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

    if not status:
        print("Frame is not been captured. Exiting...")
        raise Exception("Frame not captured. Is camera connected?")
    return frame


def load_labels():
    """Load the labels from the json file that correspond to the model outputs"""
    labels_filename = os.path.join(MODEL_PATHNAME, "student_recognition_labels.json")
    labels_file = open(labels_filename)
    return json.load(labels_file)


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


def capture_and_predict():
    """Grab an image and run it through the ML model

    Returns: predicted_name, certainty  where certainty is a value between 0 and 1.0
    """
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
    certainty = float(prediction[highest_prediction_index])
    print(
        "Prediction %d %s  Certainty: %0.2f"
        % (
            highest_prediction_index,
            student_labels[highest_prediction_index],
            certainty,
        )
    )
    return student_labels[highest_prediction_index], certainty


#####
#
student_labels = load_labels()

####################################################################
# Setup OpenCV for reading from the camera
#
camera = cv.VideoCapture(0)

# Important: Turn off the buffer on the camera. Otherwise, you get stale images
camera.set(cv.CAP_PROP_BUFFERSIZE, 1)


# Create and display the main UI
window = build_window()
set_ui_state(window, "WAITING")


#####################################################################
# Initialize the Machine Learning model. This takes some time (about 20 seconds)
load_model()


try:
    main_loop()
except BaseException as e:
    print("Exiting due to %s" % str(e))
# When everything done, release resources.
window.close()
camera.release()
if proximity_sensor is not None:
    proximity_sensor.deinit()