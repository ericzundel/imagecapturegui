"""Tests a UI idea to capture images from a camera and classify them.

This code was intended to run on a Raspberry Pi with a webcam and an
HC-SR04 ultrasonic sensor wired to GPIO 4 and GPIO 17

It presents a simple GUI that helps users classify images from the camera.
"""
import json
import os
import time
from datetime import datetime

# import numpy as np
import cv2 as cv
import PySimpleGUI as sg

DEFAULT_FONT = ("Any", 16)
LIST_HEIGHT = 14
face_choices_file_path = "face_choices.json"
rootfolder = "."
NUM_IMAGES_TO_CAPTURE = 3
# When taking multiple images, wait this many seconds between
TIME_BETWEEN_CAPTURES = .25

DISPLAY_IMAGE_WIDTH = 130
DISPLAY_IMAGE_HEIGHT = 130

# The GUI won't snap another picture unless this timer expires
WAIT_FOR_NAMING_SECS = 15


####################################################################
# Setup the Ultrasonic Sensor as a proximity sensor
# (Requires extra hardware - only works on Raspberry Pi)
#
try:
    from proximity_sensor import ProximitySensor
    proximity_sensor = ProximitySensor(echo_pin=17, trigger_pin=4, debug=True)
    print("Proximity Sensor initialized")
except BaseException:
    # It's OK, probably not running on Raspberry Pi
    proximity_sensor = None
    print("No proximity sensor detected")
    
####################################################################
# Setup OpenCV for reading from the camera
#
camera = cv.VideoCapture(0)

# Important: Turn off the buffer on the camera. Otherwise, you get stale images
camera.set(cv.CAP_PROP_BUFFERSIZE, 1)

####################################################################
# GUI Setup

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
    
def pin_image(val):
    """ Wrap the Image in a sg.pin element to make it small when invisible.

    Returns: sg.Element to add to a layout
    """
    # https://stackoverflow.com/questions/72970201
    return sg.pin(sg.Image(size=(5, 5), key="-IMAGE%d-" % val,
                    expand_x=True, expand_y=True))

def build_window(list_values):
    """Builds the user interface and pops it up on the screen.

    Returns: sg.Window object
    """
    left_column = sg.Column([
            [sg.Text(size=(18, 1), key="-STATUS-", font=DEFAULT_FONT)],
            [sg.Button("Manual Capture", key="-CAPTURE-", font=DEFAULT_FONT)],
            [pin_image(0)],
            [pin_image(1)],
            [pin_image(2)]
        ], key="-LEFT_COLUMN-", expand_x=True, expand_y=True)
    right_column = sg.Column([
            [sg.Listbox(list_values, size=(20, LIST_HEIGHT), enable_events=True,
                        key="-LIST-", font=DEFAULT_FONT)],
            [sg.Button("Cancel",  key="-CANCEL-", font=DEFAULT_FONT)],
        ], key='-RIGHT_COLUMN-', visible=False, expand_x=True, expand_y=True)
    # Push and VPush elements help UI to center when the window is maximized    
    layout = [[sg.VPush()],
              [sg.Push(), left_column, right_column, sg.Push()],
              [sg.VPush()]]
    window = sg.Window("Face Image Capture", layout, finalize=True,
                        resizable=True)
    # Doing this makes the app take up the whole screen
    window.maximize()
    return window

##########################################################################
# Image handling
# 
def save_images(images, choice):
    """Given an array of CV2 Images and a choice, save to PNG files.

    Returns: None
    """
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
        print("Wrote %s" % (filename))
        count = count + 1

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
        # Check if we get the frame or not
        if not status:
            print("Frame is not been captured. Exiting...")
            raise Exception("Frame not captured. Is camera connected?")
        images.append(frame)
        if count < NUM_IMAGES_TO_CAPTURE:
            time.sleep(TIME_BETWEEN_CAPTURES)
    #cv.destroyAllWindows()
    return images

#############################################################
# GUI Runtime actions
#
def set_ui_state(window, state):
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

    if (state == 'WAITING'):
        # Hide images and right column
        # Show manual capture button
        window["-STATUS-"].update("Waiting to Capture")
        window["-CAPTURE-"].update(visible=True)
        for i in range(3):
            window["-IMAGE%d-" % i].update(size=(0, 0), data=None, visible=False)
        window["-RIGHT_COLUMN-"].update(visible=False)
        window["-LEFT_COLUMN-"].expand(True, True)

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
    """Retrieve the selected value as a scalar, not a one item list.

    Returns: string with the displayed value selected in the sg.List()
    """
    if value_list is None:
        raise Exception("Whoops, something went wrong in retrieving value from event")
    return value_list[0]


def display_image_in_ui(image, ui_key):
    """Given an OpenCV image, display it in the UI in the element by ui_key.

    image: OpenCV Image buffer
    
    ui_key: the key for an sg.Image element in the UI layout.
    
    Returns: None
    """
    # Resize the image to fit
    resized = cv.resize(image, (DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT))
    img_bytes = cv.imencode('.png', resized)[1].tobytes()
    window[ui_key].update(data=img_bytes)

def do_capture_images():
    """Updates the UI for capturing images and pulls them from the camera.

    Returns: Array of OpenCV Images.
    """
    
    images = capture_images()
    for i in range(len(images)):
        display_image_in_ui(images[i], "-IMAGE%d-" % i)

    return images

def check_proximity_sensor():
    """Conditionally checks the proximity sensor.

    This code assumes that if the proximity sensor was detected, the
    library has been initialized and an instance of ProximitySensor set
    in a global variable.
    
    Returns: None
    """
    triggered = False
    if proximity_sensor != None:
        triggered = proximity_sensor.is_triggered()
    return triggered

# ###########################################################################
# UI Event Loop
#
# This loop executes until someone closes the main window.
#
def main_loop():
    # last_captured_images: Keep this variable to remember the last set of images captured
    # from the camera.  Clear this variable after saving them to disk.
    last_captured_images = []

    # last_captured_image_time: Used to keep from rapidly capturing images along with WAIT_FOR_NAMING_SECS
    # to give the user a chance to use the UI.
    last_captured_image_time = 0
    while True:

        # Check for a trigger 4x a second
        event, values = window.read(timeout=250)
        
        # Every time something happens in the UI, it returns an event.
        # By decoding this event you can figure out what happened and take
        # an action.
        if event in (sg.WIN_CLOSED, "Exit"):  # always check for closed window
            break

        # If the user doesn't want to classify this image, the cancel button
        # will clear out the state.
        if event == "-CANCEL-":
            last_captured_images = []
            last_captured_image_time = 0
            set_ui_state(window, 'WAITING')

        # Check to see if we are to capture new images by checking the
        # proximity sensor hardware or if the button was pressed
        if check_proximity_sensor() or event == "-CAPTURE-":
            remaining_secs = WAIT_FOR_NAMING_SECS - (time.monotonic() - last_captured_image_time)
            if (event != "-CAPTURE-"
                and last_captured_images != []
                and (remaining_secs > 0)):
                print("Waiting for %d secs for timeout before automatically capturing again." %
                      remaining_secs)
            else:
                set_ui_state(window, 'CAPTURING')
                last_captured_images = do_capture_images()
                last_captured_image_time = time.monotonic()        
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
                # Save the stored images to disk
                save_images(last_captured_images, choice)
                last_captured_images = []
                last_captured_image_time = 0
                set_ui_state(window, 'WAITING')

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

try:
    main_loop()
except BaseException as e:
    print("Exiting due to %s" % str(e))

# When everything done, release resources. 
window.close()
camera.release()
if proximity_sensor != None:
    proximity_sensor.deinit()

