"""Tests a UI idea to display faces in a window"""
import json
import os
import time
from datetime import datetime

# import numpy as np
import cv2 as cv
import PySimpleGUI as sg


face_choices_file_path = "face_choices.json"
rootfolder = "."
NUM_IMAGES_TO_CAPTURE = 3
DISPLAY_IMAGE_WIDTH = 150
DISPLAY_IMAGE_HEIGHT = 150

####################################################################
# Setup OpenCV for reading from the camera
#
camera = cv.VideoCapture(0)

def format_choice(choice_elem):
    """Format one element of the JSON array for display.

    Concatenate the first_name and last_name fields in the json array.
    This means that each entry in the JSON file must have a unique
    first name/last name combination.
    """
    return "%s %s" % (choice_elem["first_name"], choice_elem["last_name"])


def read_face_choices():
    """Read the JSON file and process it, checking for errors"""
    with open(face_choices_file_path, "r") as file:
        try:
            face_choices = json.load(file)
        except Exception as ex:
            print(
                ">>>Looks like something went wrong loading %s. Is it valid JSON?",
                (face_choices_file_path),
            )
            raise ex

        # Now, loaded_dict contains the dictionary from the file
        # print(face_choices)
        return face_choices


def build_window(list_values):
    """Builds the user interface and pops it up on the screen"""
    left_column = sg.Column([
            [sg.Text(size=(18, 1), text="WAITING", key="-STATUS-")],
            [sg.Button("Manual Capture", key="-CAPTURE-")],
            [sg.Image(size=(5, 5), key="-IMAGE0-", expand_x=True, expand_y=True)],
            [sg.Image(size=(5, 5), key="-IMAGE1-", expand_x=True, expand_y=True)],
            [sg.Image(size=(5, 5), key="-IMAGE2-", expand_x=True, expand_y=True)],
    ], key="-LEFT_COLUMN-", expand_x=True, expand_y=True)
    right_column = sg.Column([
            [sg.Listbox(list_values, size=(20, 30), enable_events=True,
                        key="-LIST-")],
            [sg.Button("Cancel",  key="-CANCEL-")],
    ], key='-RIGHT_COLUMN-', visible=False, expand_x=True, expand_y=True)
    layout = [
        [left_column, right_column],
    ]
    window = sg.Window("Face Image Capture", layout, finalize=True)
    return window


def set_ui_state(window, state):
    """Set the UI into a specified state.

    state: one of ['WAITING', 'CAPTURING', 'NAMING']

    Returns: None
    """

    if (state == 'WAITING'):
        # Hide images and right column
        # Show manual capture button
        window["-STATUS-"].update("Waiting to Capture")
        window["-CAPTURE-"].update(visible=True)
        for i in range(3):
            window["-IMAGE%d-" % i].update(size=(0, 0), data=None, visible=False)
        window["-RIGHT_COLUMN-"].update(visible=False)

    elif (state == 'CAPTURING'):
        # Hide Manual capture button
        window["-STATUS-"].update("Choose a Label")
        window["-CAPTURE-"].update(visible=False)
        for i in range(3):
            window["-IMAGE%d-" % i].update(size=(0, 0), data=None, visible=False)

    elif (state == 'NAMING'):
        # Turn on the right column
        window["-STATUS-"].update("Choose a Label")
        window["-RIGHT_COLUMN-"].update(visible=True)
        window["-CAPTURE-"].update(visible=False)
        for i in range(3):
            window["-IMAGE%d-" % i].update(data=None, visible=True)
    else:
        raise RuntimeError("Invalid state %s" % state)

def get_selected_value(value_list):
    """Retrieve the selected value as a scalar, not a one item list"""
    if value_list is None:
        raise Exception("Whoops, something went wrong in retrieving value from event")
    return value_list[0]


def save_images(images, choice):
    """Given an array of CV2 Images and a choice, save to PNG files"""
    directory = os.path.join(
        "images", "%s%s" % (choice["first_name"], choice["last_name"])
    )
    print("Capturing images for %s in dir %s" % (format_choice(choice), directory))
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
        gray = cv.resize(gray, (400, 400))
        # Store the image to specific label folder
        filename = "%s/img%s-%d.png" % (directory, timestamp, count)
        cv.imwrite(filename, gray)
        count = count + 1
        print("Wrote %s" % (filename))
        count = count + 1

def capture_images():
    """Captures NUM_IMAGES_TO_CAPTURE from the camera

    Returns: Array of image buffers
    """
    images = []
    directory = os.path.join("images", "UNNAMED")
    print("Capturing images in dir %s" % (directory))

    #
    # Call OpenCV to capture from the camera
    #
    if not os.path.exists(directory):
        os.mkdir(directory)

    count = 0
    while count < NUM_IMAGES_TO_CAPTURE:
        count = count + 1
        # Read returns two values one is the exit code and other is the frame
        status, frame = camera.read()
        # Check if we get the frame or not
        if not status:
            print("Frame is not been captured. Exiting...")
            raise Exception("Frame not captured")
        images.append(frame)
        if count < NUM_IMAGES_TO_CAPTURE:
            print("Wait to take another image...")
            time.sleep(0.25)
    cv.destroyAllWindows()
    return images

def display_image_in_ui(image, ui_key):
    # Resize the image to fit
    resized = cv.resize(image, (DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT))
    img_bytes = cv.imencode('.png', resized)[1].tobytes()
    window[ui_key].update(data=img_bytes)

def do_capture_images():
    window["-STATUS-"].update("Capturing")
    images = capture_images()
    for i in range(len(images)):
        display_image_in_ui(images[i], "-IMAGE%d-" % i)
    return images

# ###################################################################
# Setup the User Interface
#

# Read the JSON file in
face_choices = read_face_choices()

# Format the names in the file for display in a listbox
names = sorted([format_choice(elem) for elem in face_choices])
print("List of names found in JSON file is:", names)

# Create and display the main UI
window = build_window(names)
set_ui_state(window, 'WAITING')
last_captured_images = []

# ###########################################################################
# UI Event Loop
#
# This loop executes until someone closes the main window.
#
while True:
    start_ns = time.monotonic_ns()

    # Check for a trigger about every 10 milliseconds
    event, values = window.read(timeout=10)

    # Every time something happens in the UI, it returns an event.
    # By decoding this event you can figure out what happened and take
    # an action.
    if event in (sg.WIN_CLOSED, "Exit"):  # always check for closed window
        break

    if event == "-CANCEL-":
        last_captured_images = []
        set_ui_state(window, 'WAITING')

    if event == "-CAPTURE-":
        set_ui_state(window, 'CAPTURING')
        last_captured_images = do_capture_images()
        set_ui_state(window, 'NAMING')

    # if a list item is clicked on, the following code gest triggered
    if event == "-LIST-" and len(values["-LIST-"]):
        selected_value = get_selected_value(values["-LIST-"])
        choice_list = [
            choice for choice in face_choices if format_choice(choice) == selected_value
        ]

        if choice_list is None:
            sg.popup("Whoops, something went wrong when retrieving element from list")
        elif last_captured_images is []:
            sg.popup("Whoops, no images captured")
        else:
            # Now we can get the original object back from the json file
            choice = choice_list[0]
            ####
            # Do something
            save_images(last_captured_images, choice)
            last_captured_images = []
            set_ui_state(window, 'WAITING')

window.close()

# When everything done, release the capture
camera.release()

