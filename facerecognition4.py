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
import PySimpleGUI as sg
from gtts import gTTS

# This weird import code is so we can support both the full
# Tensorflow library (linux, windows) and Tensorflow Lite
# (linux).

# Besides the fact that it is good for testing to try both versions,
# Currently, Google doesn't make tensorflow Lite
# binaries available for Windows.

tensorflow_type = None
names = []
models = []
interpreters = []
vision = None

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

###################################################################
# Constants

FACE_RECOGNITION_IMAGE_WIDTH = 100
FACE_RECOGNITION_IMAGE_HEIGHT = 100

# Local path to find the model files
MODEL_PATHNAME_BASE = "./2024model/"

model_dict = {
    "Rhyland's Model": "Rhyland_student_recognition_2024_32bit",
    "Laila's Model" : "Laila_student_recognition_2024_32bit",
    "Justin Brown's Model" : "Justin_Brown_student_recognition_2024_32bit",
    "Kenadie's Model" : "Kenadie_student_recognition_2024_32bit"
    }

LABEL_FILENAME = "student_recognition_labels.json"

TEST_IMAGE1 = os.path.join(MODEL_PATHNAME_BASE, "donald_test.png")
TEST_IMAGE2 = os.path.join(MODEL_PATHNAME_BASE, "laila_test.png")

DEFAULT_FONT = ("Any", 16)
SMALLER_FONT = ("Any", 12)
LIST_HEIGHT = 12  # Number of rows in listbox element
LIST_WIDTH = 20  # Characters wide for listbox element

DISPLAY_IMAGE_WIDTH = 120  # Size of image when displayed on screen
DISPLAY_IMAGE_HEIGHT = 120

DISPLAY_TIMEOUT_SECS = 12


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
    global names
    global models
    global interpreters

    names = list(model_dict.keys())
    if tensorflow_type == "FULL":
        print("Initializing Tensorflow Version" + tf.__version__)
        for name in names:    
            model = tf.keras.models.load_model(
                os.path.join(MODEL_PATHNAME_BASE, "%s.tf" % (model_dict[name]))
            )
            models.append(model)
            interpreters.append(None)
            # Sanity check the model after loading
            #model.summary()            

    elif tensorflow_type == "LITE":
        print("Initializing Tensorflow Lite")
        for name in names:                              
            # Load the TFLite model
            interpreter = tflite_runtime.Interpreter(
                model_path=os.path.join(
                    MODEL_PATHNAME_BASE, "%s.tflite" % (model_dict[name])
                )
            )
            interpreter.allocate_tensors()
            interpreters.append(interpreter)
            models.append(None)

def tensor_from_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (FACE_RECOGNITION_IMAGE_WIDTH, FACE_RECOGNITION_IMAGE_HEIGHT))
    arr = my_img_to_arr(img) / 255.0

    print("Shape of array is: ")
    print(arr.shape)
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


def pretty_print_predictions(model_name, prediction, labels):
    print("Predicts for Model: %s" % (model_name))
    prediction_arr = prediction
    for i, prob in enumerate(prediction_arr):
        try:
            print("%37s: %2.2f%%" % (labels[i], prob * 100.0))
        except IndexError as ex:
            print("Index out of range: %d" % i)
            print(ex)


def my_img_to_arr(image):
    # return np.expand_dims(image, axis=0)
    if tensorflow_type == "FULL":
        return tf.keras.utils.img_to_array(image).reshape(FACE_RECOGNITION_IMAGE_WIDTH * FACE_RECOGNITION_IMAGE_HEIGHT)
    elif tensorflow_type == "LITE":
        return np.asarray(image, dtype=np.float32).reshape(FACE_RECOGNITION_IMAGE_WIDTH * FACE_RECOGNITION_IMAGE_HEIGHT)


#############################################################################
#  GUI code
def build_window():
    """Builds the user interface and pops it up on the screen.

    Returns: sg.Window object
    """
    left_column = sg.Column(
        [
            [sg.Text("", size=(18, 1), key="-STATUS-", font=DEFAULT_FONT)],
            [sg.pin(sg.Button("Manual Capture", key="-CAPTURE-", font=DEFAULT_FONT))],
            [sg.Text()],  # vertical spacer
            [sg.Text()],  # vertical spacer
            [sg.Text()],  # vertical spacer
            [sg.Button("Test Donald", key="-TEST_IMAGE1-", font=("Any", 10))],
            [sg.Button("Test Laila", key="-TEST_IMAGE2-", font=("Any", 10))],
            [sg.Button("Exit", font=("Any", 6))],
        ],
        key="-LEFT_COLUMN-",
    )
    right_column = sg.Column(
        [
            [
                sg.Text("Model: ", font=SMALLER_FONT),
                sg.Text(key="-MODEL_NAME1-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Name: ", font=SMALLER_FONT),
                sg.Text(key="-FACE_NAME1-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Certainty: ", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY1-", font=SMALLER_FONT),
            ],
            [
                sg.Text("Model: ", font=SMALLER_FONT),
                sg.Text(key="-MODEL_NAME2-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Name: ", font=SMALLER_FONT),
                sg.Text(key="-FACE_NAME2-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Certainty: ", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY2-", font=SMALLER_FONT),
            ],            
            [
                sg.Text("Model: ", font=SMALLER_FONT),
                sg.Text(key="-MODEL_NAME3-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Name: ", font=SMALLER_FONT),
                sg.Text(key="-FACE_NAME3-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Certainty: ", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY3-", font=SMALLER_FONT),
            ],
            [
                sg.Text("Model: ", font=SMALLER_FONT),
                sg.Text(key="-MODEL_NAME4-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Name: ", font=SMALLER_FONT),
                sg.Text(key="-FACE_NAME4-", font=SMALLER_FONT),
            ],
            [
                sg.Text("  Certainty: ", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY4-", font=SMALLER_FONT),
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


def set_ui_state(window, state, face_names=None, certainties=None):
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
        
        window["-MODEL_NAME1-"].update(names[0])
        window["-FACE_NAME1-"].update(face_names[0])
        window["-CERTAINTY1-"].update("%2f" % (certainties[0] * 100.0))
        
        window["-MODEL_NAME2-"].update(names[1])
        window["-FACE_NAME2-"].update(face_names[1])
        window["-CERTAINTY2-"].update("%2f" % (certainties[1] * 100.0))
        
        window["-MODEL_NAME3-"].update(names[2])
        window["-FACE_NAME3-"].update(face_names[2])
        window["-CERTAINTY3-"].update("%2f" % (certainties[2] * 100.0))
        
        window["-MODEL_NAME4-"].update(names[3])
        window["-FACE_NAME4-"].update(face_names[3])
        window["-CERTAINTY4-"].update("%2f" % (certainties[3] * 100.0))
        
        window["-CAPTURE-"].update(visible=False)
    else:
        raise RuntimeError("Invalid state %s" % state)


def main_loop(labels):
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
        if (
            check_button()
            or event == "-CAPTURE-"
            or event == "-TEST_IMAGE1-"
            or event == "-TEST_IMAGE2-"
        ):
            set_ui_state(window, "CAPTURING")
            last_captured_image_time = time.monotonic()
            # FOR DEBUGGING
            # Try some test images
            if event == "-TEST_IMAGE1-":
                predicted_names, certainties = test_and_predict(TEST_IMAGE1,
                                                                labels)
            elif event == "-TEST_IMAGE2-":
                predicted_names, certainties = test_and_predict(TEST_IMAGE2,
                                                                labels)
            else:
                predicted_names, certainties = capture_and_predict(labels)
            set_ui_state(window, "NAMING", face_names=predicted_names,
                         certainties=certainties)


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


def say_names(predicted_names):
    # First, uniquify the list
    predicted_names = list(set(predicted_names))

    # Customize the prompt to 
    if (len(predicted_names) == 1):
        first_name = predicted_names[0].split(sep="_")[0]
        text_to_speech("Hello, %s" % (first_name))
    elif (len(predicted_names) == 2):
        first_name1 = predicted_names[0].split(sep="_")[0]
        first_name2 = predicted_names[1].split(sep="_")[0]
        text_to_speech("Are you %s or %s?" % (first_name1, first_name2))
    elif (len(predicted_names) == 3):
        first_name1 = predicted_names[0].split(sep="_")[0]
        first_name2 = predicted_names[1].split(sep="_")[0]
        first_name3 = predicted_names[2].split(sep="_")[0]
        text_to_speech("Are you %s, %s, or %s?" % (first_name1, first_name2, first_name3))
    else:
        text_to_speech("I am very confused")
def do_predict(img, labels):
    tensor = tensor_from_image(img)

    print("tensor is:")
    print(tensor)

    # Run the image through the model to see which output it predicts
    model_predicted_names = []
    certainties = []
    for i in range(len(names)):
        name = names[i]
        model = models[i]
        interpreter = interpreters[i]
        prediction = predict(model, interpreter, tensor)

        # import pdb; pdb.set_trace()
        pretty_print_predictions(name, prediction, labels)

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
        predicted_name = labels[highest_prediction_index]
        
        model_predicted_names.append(predicted_name)
        certainties.append(certainty)

    say_names(model_predicted_names)
    return model_predicted_names, certainties


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
window = build_window()
set_ui_state(window, "WAITING")


#####################################################################
# Initialize the Machine Learning model. This takes some time (about 20 seconds)
load_model()

try:
    main_loop(labels)
except BaseException as e:
    print("Exiting due to %s " % str(e))
    print(traceback.format_exc())

# When everything done, release resources.
window.close()
camera.release()
if proximity_sensor is not None:
    proximity_sensor.deinit()
