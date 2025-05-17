"""GUI w/ data collection and model predicton using two hardware buttons

Merges facerecognition*.py and imagecapture.py

Repo is https://github.com/ericzundel/imagecapturegui

Hardware button 1: Capture Data
Hardware button 2: Predict based on canned demo data

See README.md for how to connect the buttons to the Raspberry Pi GPIO header.

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
from datetime import datetime

import numpy as np
import cv2 as cv

# import matplotlib.pyplot as plt
import PySimpleGUI as sg
from gtts import gTTS, gTTSError

# This weird import code is so we can support both the full
# Tensorflow library (linux, windows) and Tensorflow Lite
# (linux).

# Besides the fact that it is good for testing to try both versions,
# Currently, Google doesn't make tensorflow Lite
# binaries available for Windows.

tensorflow_type = None
names = []
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
MODEL_PATHNAME_BASE = "./2025model/"

model_dict = {
    "Regie's Model": "regie_ingram_2025",
    "Jai's Model": "jai_bazawule_2025",
    "Penny's Model": "penny_dunn_2025",
    "Michael's Model": "michael_rashad_2025",
}

LABEL_FILENAME = "student_recognition_labels.json"

# Data to be returned by get_test_image()
test_image_db = (
    (os.path.join(MODEL_PATHNAME_BASE, "rhyland_test.png"), "Regie"),
    (os.path.join(MODEL_PATHNAME_BASE, "alexandra_test.png"), "Jai"),
    (os.path.join(MODEL_PATHNAME_BASE, "donald_test.png"), "Penny"),
    (os.path.join(MODEL_PATHNAME_BASE, "laila_test.png"), "Michael"),
    # TODO(ericzundel): Add more images here
)

# Keep track of the last image tested so we can rotate through them.
next_test_image = 0

DEFAULT_FONT = ("Any", 16)
SMALLER_FONT = ("Any", 12)
LIST_HEIGHT = 12  # Number of rows in listbox element
LIST_WIDTH = 20  # Characters wide for listbox element
FACE_CHOICES_FILE_PATH = "face_choices.json"
NUM_IMAGES_TO_CAPTURE = 3  # Number of frames to capture from the camera
TIME_BETWEEN_CAPTURES = 0.2  # Wait this many seconds between capturing

DISPLAY_IMAGE_WIDTH = 300  # Size of image when displayed on screen
DISPLAY_IMAGE_HEIGHT = 300

DISPLAY_TIMEOUT_SECS = 20  # Increased for Hill Demo Was 12 for classroom.

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

    capture_button_pin = 21
    predict_button_pin = 20

    # Setup IO pin for buttons
    GPIO.setup(capture_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(predict_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

############################################################################
# ML code


def load_model():
    global names
    global interpreters

    names = list(model_dict.keys())
    for name in names:
        model_path = os.path.join(
            MODEL_PATHNAME_BASE,
            "%s.tflite" %
            (model_dict[name]))
        interpreter = None
        if tensorflow_type == "FULL":
            interpreter = tf.lite.Interpreter(model_path=model_path)
        elif tensorflow_type == "LITE":
            # Load the TFLite model
            interpreter = tflite_runtime.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        interpreters.append(interpreter)


def tensor_from_image(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (FACE_RECOGNITION_IMAGE_WIDTH,
                    FACE_RECOGNITION_IMAGE_HEIGHT))
    arr = my_img_to_arr(img) / 255.0

    print("Shape of array is: ")
    print(arr.shape)
    # plt.imshow(arr)
    # plt.show()

    return arr


def predict(interpreter, tensor):
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
    return np.asarray(image, dtype=np.float32).reshape(
        FACE_RECOGNITION_IMAGE_WIDTH * FACE_RECOGNITION_IMAGE_HEIGHT
    )


#############################################################################
#  GUI code


def format_choice(choice_elem):
    """Format one element of the JSON array for display.

    Concatenate the first_name and last_name fields in the json array.
    This means that each entry in the JSON file must have a unique
    first name/last name combination.

    Returns: string with first name and last name separated by a space.
    """
    return "%s %s" % (choice_elem["first_name"], choice_elem["last_name"])


def read_face_choices():
    """Read the JSON file and process it, checking for errors.

    Returns: dictionary containing the data in the json file.
    """
    with open(FACE_CHOICES_FILE_PATH, "r") as file:
        try:
            face_choices = json.load(file)
        except Exception as ex:
            print(
                ">>>Looks like something went wrong loading %s. Is it valid JSON?",
                (FACE_CHOICES_FILE_PATH),
            )
            raise ex
        # Now, loaded_dict contains the dictionary from the file
        # print(face_choices)
        return face_choices


def build_window(list_values):
    """Builds the user interface and pops it up on the screen.

    Returns: sg.Window object
    """
    left_column = sg.Column(
        [
            [sg.Text("", size=(28, 1), key="-STATUS-", font=DEFAULT_FONT)],
            [sg.Text()],  # vertical spacer
            [
                sg.pin(
                    sg.Image(
                        size=(
                            5,
                            5),
                        key="-IMAGE-",
                        expand_x=True,
                        expand_y=True),
                ),
            ],
            [
                sg.pin(
                    sg.Text(
                        "",
                        size=(
                            18,
                            1),
                        key="-EXPECTED-LABEL-",
                        font=DEFAULT_FONT)
                )
            ],
            #  [sg.Text()],  # vertical spacer
            [sg.pin(sg.Button("Capture", key="-CAPTURE-", font=DEFAULT_FONT))],
            [sg.Button("Predict", key="-PREDICT-", font=DEFAULT_FONT)],
            [sg.Button("Exit", font=("Any", 6))],
        ],
        key="-LEFT_COLUMN-",
    )
    predict_column = sg.Column(
        [
            [
                sg.Text(key="-MODEL_NAME1-", font=SMALLER_FONT),
            ],
            [
                sg.Text(),  # horizontal spacer
                sg.Text(key="-FACE_NAME1-", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY1-", font=SMALLER_FONT),
            ],
            [sg.Text()],  # vertical spacer
            [
                sg.Text(key="-MODEL_NAME2-", font=SMALLER_FONT),
            ],
            [
                sg.Text(),  # horizontal spacer
                sg.Text(key="-FACE_NAME2-", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY2-", font=SMALLER_FONT),
            ],
            [sg.Text()],  # vertical spacer
            [
                sg.Text(key="-MODEL_NAME3-", font=SMALLER_FONT),
            ],
            [
                sg.Text(),  # horizontal spacer
                sg.Text(key="-FACE_NAME3-", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY3-", font=SMALLER_FONT),
            ],
            [sg.Text()],  # vertical spacer
            [
                sg.Text(key="-MODEL_NAME4-", font=SMALLER_FONT),
            ],
            [
                sg.Text(),  # horizonatal spacer
                sg.Text(key="-FACE_NAME4-", font=SMALLER_FONT),
                sg.Text(key="-CERTAINTY4-", font=SMALLER_FONT),
            ],
            [sg.Text()],  # vertical spacer
            [sg.Button("Cancel", key="-CANCEL-", font=DEFAULT_FONT)],
        ],
        key="-PREDICT_COLUMN-",
        visible=False,
    )

    naming_column = sg.Column(
        [
            [
                sg.Listbox(
                    list_values,
                    size=(LIST_WIDTH, LIST_HEIGHT),
                    enable_events=True,
                    key="-LIST-",
                    font=("Any", 18),
                    sbar_width=30,
                    sbar_arrow_width=30,
                )
            ],
            [sg.Button("Cancel", key="-CANCEL2-", font=DEFAULT_FONT)],
        ],
        key="-NAMING_COLUMN-",
        visible=False,
    )
    # Push and VPush elements help UI to center when the window is maximized
    layout = [
        [sg.VPush()],
        [
            sg.Push(),
            sg.Column(
                [
                    [
                        left_column,
                        sg.pin(predict_column),
                        sg.pin(naming_column),
                    ]
                ]
            ),
            sg.Push(),
        ],
        [sg.VPush()],
    ]
    window = sg.Window(
        "Face Image Capture",
        layout,
        finalize=True,
        resizable=True)
    # Doing this makes the app take up the whole screen
    window.maximize()
    return window


def get_selected_value(value_list):
    """Retrieve the selected value as a scalar, not a one item list.

    Returns: string with the displayed value selected in the sg.List()
    """
    if value_list is None:
        raise Exception(
            "Whoops, something went wrong in retrieving value from event")
    return value_list[0]


def display_image_in_ui(image, ui_key):
    """Given an OpenCV image, display it in the UI in the element by ui_key.

    image: OpenCV Image buffer

    ui_key: the key for an sg.Image element in the UI layout.

    Returns: None
    """
    # Resize the image to fit
    resized = cv.resize(image, (DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT))
    img_bytes = cv.imencode(".png", resized)[1].tobytes()
    window[ui_key].update(data=img_bytes, visible=True)


def set_ui_state(
    window, state, face_names=None, certainties=None, image=None, expected_label=None
):
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
        window["-PREDICT-"].update(visible=True)
        window["-IMAGE-"].update(size=(0, 0), data=None, visible=False)
        window["-EXPECTED-LABEL-"].update("", visible=False)
        window["-PREDICT_COLUMN-"].update(visible=False)
        window["-NAMING_COLUMN-"].update(visible=False)
        window["-LEFT_COLUMN-"].expand(True, True)
        window["-CANCEL-"].update(visible=True)
        window["-CANCEL2-"].update(visible=False)
    elif state == "CAPTURING":
        # Hide Manual capture button
        window["-STATUS-"].update("Running ML prediction")
        window["-IMAGE-"].update(size=(0, 0), data=None, visible=False)
        window["-EXPECTED-LABEL-"].update("", visible=False)
        window["-PREDICT_COLUMN-"].update(visible=False)
        window["-NAMING_COLUMN-"].update(visible=False)
        window["-CAPTURE-"].update(visible=False)
        window["-PREDICT-"].update(visible=False)
        window["-CANCEL-"].update(visible=True)
        window["-CANCEL2-"].update(visible=False)
    elif state == "PREDICTING":
        # Turn on the prediction column
        window["-STATUS-"].update("Test Image")
        display_image_in_ui(image, "-IMAGE-")
        if expected_label is not None:
            expected_label = "Expected: %s" % expected_label
        window["-EXPECTED-LABEL-"].update(expected_label, visible=True)
        window["-PREDICT_COLUMN-"].update(visible=True)
        window["-NAMING_COLUMN-"].update(visible=False)

        window["-MODEL_NAME1-"].update(names[0])
        window["-FACE_NAME1-"].update(face_names[0])
        window["-CERTAINTY1-"].update("%.0f%%" % (certainties[0] * 100.0))

        window["-MODEL_NAME2-"].update(names[1])
        window["-FACE_NAME2-"].update(face_names[1])
        window["-CERTAINTY2-"].update("%.0f%%" % (certainties[1] * 100.0))

        window["-MODEL_NAME3-"].update(names[2])
        window["-FACE_NAME3-"].update(face_names[2])
        window["-CERTAINTY3-"].update("%.0f%%" % (certainties[2] * 100.0))

        window["-MODEL_NAME4-"].update(names[3])
        window["-FACE_NAME4-"].update(face_names[3])
        window["-CERTAINTY4-"].update("%.0f%%" % (certainties[3] * 100.0))

        window["-CAPTURE-"].update(visible=False)
        window["-PREDICT-"].update(visible=False)
        window["-CANCEL-"].update(visible=True)
        window["-CANCEL2-"].update(visible=False)
    elif state == "NAMING":

        window["-STATUS-"].update("Label (Classify) your image")
        window["-CAPTURE-"].update(visible=False)
        if image is None:
            raise RuntimeError("No image passed. ")
        display_image_in_ui(image, "-IMAGE-")
        window["-PREDICT_COLUMN-"].update(visible=False)
        window["-NAMING_COLUMN-"].update(visible=True)
        window["-CANCEL-"].update(visible=False)
        window["-CANCEL2-"].update(visible=True)
    else:
        raise RuntimeError("Invalid state %s" % state)


def main_loop(labels):
    """UI Event Loop

    This loop executes until someone closes the main window.
    """

    last_captured_images = None
    last_captured_image_time = 0

    while True:
        # Check for a trigger 4x a second
        event, values = window.read(timeout=50)
        expected_label = None

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
        if event == "-CANCEL-" or event == "-CANCEL2-":
            last_captured_images = None
            last_captured_image_time = 0
            set_ui_state(window, "WAITING")

        # Check to see if we are to capture new images by checking the
        # proximity sensor hardware or if the button was pressed
        capture_pressed = check_capture_button() or event == "-CAPTURE-"
        predict_pressed = check_predict_button() or event == "-PREDICT-"
        if capture_pressed or predict_pressed:
            set_ui_state(window, "CAPTURING")
            window.read(timeout=1)  # HACK: Force the window to update
            last_captured_image_time = time.monotonic()
            captured_image = None
            if predict_pressed:
                # For demo purposes/debugging, Try some test images
                (captured_image, expected_label) = get_test_image_and_label()

                # For live device, try this:
                # captured_image = capture_image()
                # expected_label = None

                predicted_names, certainties = do_predict(
                    captured_image, labels, expected_label
                )
            else:
                last_captured_images = capture_images()

                set_ui_state(window, "NAMING", image=last_captured_images[0])

        # if a list item is clicked on, the following code gest triggered
        if event == "-LIST-" and len(values["-LIST-"]):
            selected_value = get_selected_value(values["-LIST-"])
            choice_list = [
                choice
                for choice in face_choices
                if format_choice(choice) == selected_value
            ]

            if choice_list is None:
                sg.popup(
                    "Whoops, something went wrong when retrieving element from list"
                )
            elif last_captured_images is None:
                # sg.popup("Whoops, no images captured")
                pass
            else:
                # Now we can get the original object back from the json file
                choice = choice_list[0]
                # TODO(ericzundel): Using a dialog is problematic with the Raspberry Pi
                # Windowing System. The dialog can get stuck underneath the main window
                # making the UI unresponsive.
                # Eliminating the confirmation dialog for the Hill Street Demo.
                # if confirm_choice(choice):
                if True:
                    # TODO(): Save the stored images to disk
                    save_images(last_captured_images, choice)
                    text_to_speech(
                        "Thank you for collecting and labeling your data, %s" %
                        (choice["first_name"]))
                    last_captured_images = None
                    last_captured_image_time = 0
                    set_ui_state(window, "WAITING")


###########################################################
# Other functions


def check_capture_button():
    """Check hardware button to trigger image capture from the camera. (Active Low)"""
    if GPIO is not None:
        if GPIO.input(capture_button_pin) == GPIO.LOW:
            return True
    return False


def check_predict_button():
    """Check hardware button to trigger prediction. (Active Low)"""
    if GPIO is not None:
        if GPIO.input(predict_button_pin) == GPIO.LOW:
            return True
    return False


def save_images(images, choice):
    """Given an array of CV2 Images and a choice, save to PNG files.

    Returns: None
    """
    first_last = "%s_%s" % (choice["first_name"], choice["last_name"])
    directory = os.path.join("images", first_last)
    print("Capturing images for %s in dir %s" %
          (format_choice(choice), directory))
    #
    # Call OpenCV to capture from the camera
    #
    if not os.path.exists(directory):
        os.mkdir(directory)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    count = 0
    for image in images:
        # Convert the image into gray format for fast caculation
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Resizing the image to store it
        # gray = cv.resize(gray, (400, 400))
        # Store the image to specific label folder
        filename = "%s/img%s-%d.png" % (directory, timestamp, count)
        cv.imwrite(filename, gray)
        print("Wrote %s" % (filename))
        count = count + 1


def capture_image():
    """Captures a single image from the camera

    Returns: image buffer
    """

    status, frame = camera.read()
    # HACK: Throw away the previous frame, it might be cached
    status, frame = camera.read()

    if not status:
        print("Frame is not been captured. Exiting...")
        raise Exception("Frame not captured. Is camera connected?")
    return frame


def capture_images():
    """Captures NUM_IMAGES_TO_CAPTURE from the camera

    Returns: Array of image buffers
    """
    images = []
    count = 0

    # Important! Throw the first frame away. It's a stale buffered image
    status, frame = camera.read()

    while count < NUM_IMAGES_TO_CAPTURE:
        count = count + 1
        # Read returns two values one is the exit code and other is the frame
        status, frame = camera.read()
        images.append(capture_image())
        if count < NUM_IMAGES_TO_CAPTURE:
            time.sleep(TIME_BETWEEN_CAPTURES)
    return images


def load_labels():
    """Load the labels from the json file that correspond to the model outputs"""
    labels_filename = os.path.join(MODEL_PATHNAME_BASE, LABEL_FILENAME)
    with open(labels_filename) as f:
        return json.load(f)


def confirm_choice(choice):
    name = "%s %s" % (choice["first_name"], choice["last_name"])
    result = sg.popup_ok_cancel(
        "Save for %s?" % name, keep_on_top=True, font=DEFAULT_FONT
    )
    if result == "OK":
        return True
    return False


def text_to_speech(text):
    try:
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
    except gTTSError:
        print("text to speech (gTTS) unavailable. Is the network down?")
    except Exception as e:
        print("Couldn't say: %s! %s" % (text, e))


def say_names(predicted_names):
    # First, uniquify the list
    predicted_names = list(set(predicted_names))

    # Customize the prompt to
    if len(predicted_names) == 1:
        first_name = predicted_names[0].split(sep="_")[0]
        text_to_speech("Hello, I predict you are %s" % (first_name))
    elif len(predicted_names) == 2:
        first_name1 = predicted_names[0].split(sep="_")[0]
        first_name2 = predicted_names[1].split(sep="_")[0]
        text_to_speech("Are you %s or %s?" % (first_name1, first_name2))
    elif len(predicted_names) == 3:
        first_name1 = predicted_names[0].split(sep="_")[0]
        first_name2 = predicted_names[1].split(sep="_")[0]
        first_name3 = predicted_names[2].split(sep="_")[0]
        text_to_speech(
            "Are you %s, %s, or %s?" % (first_name1, first_name2, first_name3)
        )
    else:
        text_to_speech("I am very confused")


def do_predict(img, labels, expected_label):
    tensor = tensor_from_image(img)

    print("tensor is:")
    print(tensor)

    # Run the image through the model to see which output it predicts
    model_predicted_names = []
    certainties = []
    for i in range(len(names)):
        name = names[i]
        prediction = predict(interpreters[i], tensor)

        pretty_print_predictions(name, prediction, labels)

        # Display the entry with the highest probability
        highest_prediction_index = prediction.argmax()
        certainty = float(prediction[highest_prediction_index])
        print(
            "Prediction %d %s  Certainty: %.2f"
            % (
                highest_prediction_index,
                labels[highest_prediction_index],
                certainty,
            )
        )
        predicted_name = labels[highest_prediction_index]

        model_predicted_names.append(predicted_name)
        certainties.append(certainty)

    set_ui_state(
        window,
        "PREDICTING",
        face_names=model_predicted_names,
        certainties=certainties,
        image=img,
        expected_label=expected_label,
    )

    # Force the UI to update
    window.read(timeout=1)

    say_names(model_predicted_names)
    return model_predicted_names, certainties


def get_test_image_and_label():
    """Cycle through our internal database of test images"""
    global next_test_image
    (image_path, label) = test_image_db[next_test_image]
    next_test_image = (next_test_image + 1) % len(test_image_db)
    return (cv.imread(image_path, cv.IMREAD_COLOR), label)


#####
# Load list of labels associated with the saved ML Models to display when
# predicting
labels = load_labels()

# Read the JSON file in with names to select for classifying
face_choices = read_face_choices()

# Format the names in the file for display in a listbox
names = sorted([format_choice(elem) for elem in face_choices])
print("List of names found in JSON file is:", names)

####################################################################
# Setup OpenCV for reading from the camera
#
camera = cv.VideoCapture(0)

# Important: Turn off the buffer on the camera. Otherwise, you get stale images
camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

# Create and display the main UI
window = build_window(names)
set_ui_state(window, "WAITING")


#####################################################################
# Initialize the Machine Learning model. This takes some time (about 20
# seconds)
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
