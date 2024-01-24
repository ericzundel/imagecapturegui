# See PySimpleGui/Demo_Listbox_Search_Filter.py
#
import json
import os
import time
from datetime import datetime

import cv2 as cv
import numpy as np
import PySimpleGUI as sg
from gpiozero import DistanceSensor

face_choices_file_path = 'face_choices.json'
rootfolder = "."
NUM_IMAGES_TO_CAPTURE = 3
DISTANCE_THRESHOLD = .9
ultrasonic_sensor = DistanceSensor(echo=17, trigger=4)

####################################################################
# Setup OpenCV for reading from the camera
#
camera = cv.VideoCapture(0)


def format_choice(choice_elem):
    """ Format one element of the JSON array for display.

    Concatenate the first_name and last_name fields in the json array.
    This means that each entry in the JSON file must have a unique
    first name/last name combination.
    """
    return "%s %s" % (choice_elem['first_name'], choice_elem['last_name'])

def read_face_choices():
    """Read the JSON file and process it, checking for errors"""
    with open(face_choices_file_path, 'r') as file:
        try:
            face_choices = json.load(file)
        except Exception as ex:
            print(">>>Looks like something went wrong loading %s. Is it valid JSON?",
                  (face_choices_file_path))
            raise ex
    
        # Now, loaded_dict contains the dictionary from the file
        #print(face_choices)
        return face_choices

def build_window(list_values):
    """Builds the user interface and pops it up on the screen"""
    layout = [[sg.Text('Select a name to capture images')],
          [sg.Input(size=(20, 1), enable_events=True, key='-INPUT-')],
          [sg.Listbox(list_values, size=(20, 30), enable_events=True, key='-LIST-')],
          [sg.Button('Exit')]]

    return sg.Window('Face Image Capture', layout)

def get_selected_value(value_list):
    """Retrieve the selected value as a scalar, not a one item list"""
    if (value_list is None):
        raise Exception("Whoops, something went wrong in retrieving value from event")
    
    return value_list[0]

def capture_and_save_images(choice):
    directory = os.path.join("images", "%s%s" % (choice['first_name'], choice['last_name']))
    print("Capturing images for %s in dir %s" % (format_choice(choice), directory))
    ###
    ### Call OpenCV to capture from the camera
    ###
    if not os.path.exists(directory):
        os.mkdir(directory)

    count = 0
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    while count < NUM_IMAGES_TO_CAPTURE:
        #read returns two values one is the exit code and other is the frame
        status, frame = camera.read()
        #check if we get the frame or not
        if not status:
            print("Frame is not been captured. Exiting...")
            raise Exception("Frame not captured")
        
        #convert the image into gray format for fast caculation
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #display window with gray image
        cv.imshow("Video Window",gray)
        #resizing the image to store it
        gray = cv.resize(gray, (200,200))
        #Store the image to specific label folder
        filename = '%s/img%s-%d.png' % (directory, timestamp, count)
        cv.imwrite(filename, gray)
        count=count+1
        print("Wrote %s" % (filename))
        if (count < NUM_IMAGES_TO_CAPTURE):
            print("Wait to take another image...")
            sg.popup("Captured %d of %d images. Click OK to take another image." % (count, NUM_IMAGES_TO_CAPTURE))
        cv.destroyAllWindows()

def capture_images():
    """Captures and saves Images to disk, returns an array of pathnames

    Returns: Array of paths to images
    """
    images=[]
    directory = os.path.join("images", 'UNNAMED')
    print("Capturing images in dir %s" % (directory))
    ###
    ### Call OpenCV to capture from the camera
    ###
    if not os.path.exists(directory):
        os.mkdir(directory)

    count = 0
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    while count < NUM_IMAGES_TO_CAPTURE:
        #read returns two values one is the exit code and other is the frame
        status, frame = camera.read()
        #check if we get the frame or not
        if not status:
            print("Frame is not been captured. Exiting...")
            raise Exception("Frame not captured")
        
        #convert the image into gray format for fast caculation
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #display window with gray image
        cv.imshow("Video Window",gray)
        #resizing the image to store it
        gray = cv.resize(gray, (200,200))
        #Store the image to specific label folder
        filename = '%s/img%s-%d.png' % (directory, timestamp, count)
        cv.imwrite(filename, gray)
        count=count+1
        print("Wrote %s" % (filename))
        images.append(filename)
        if (count < NUM_IMAGES_TO_CAPTURE):
            print("Wait to take another image...")
            time.sleep(.25)
        cv.destroyAllWindows()
    return images

def check_distance_sensor():
    distance = ultrasonic_sensor.distance
    print("Distance is %f" % (distance))
    if (distance < DISTANCE_THRESHOLD):        
        return True
    return False
    
############################-########################################
# Setup the User Interface
#

# Read the JSON file in
face_choices = read_face_choices()

# Format the names in the file for display in a listbox
names = sorted([format_choice(elem) for elem in face_choices])
print("List of names found in JSON file is:", names)

# Create and display the main UI
window = build_window(names)


############################################################################
# UI Event Loop
#
# This loop executes until someone closes the main window.
#
while True:
    start_ns = time.monotonic_ns()

    # Check for a trigger about every 50 milliseconds
    event, values = window.read(timeout=10)
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1000000
    #if (elapsed_ms > 5):
    if True:
        print("Elapsed ms=%d" % (elapsed_ms))
        presence = check_distance_sensor()
        if presence:
            image_files = capture_images()
            sg.popup('Tripped! %f\nSaved to %s' % (ultrasonic_sensor.distance, "\n".join(image_files)))
            

    # Every time something happens in the UI, it returns an event.
    # By decoding this event you can figure out what happened and take
    # an action.
    if event in (sg.WIN_CLOSED, 'Exit'):                # always check for closed window
        break

    # Someone entered date in the search field
    if values['-INPUT-'] != '': 
        search = values['-INPUT-']
        new_values = [x for x in names if search in x]  # do the filtering
        window['-LIST-'].update(new_values)     # display in the listbox

    # The search field has been cleared.
    else:
        # display original unfiltered list
        window['-LIST-'].update(names)
        
    # if a list item is clicked on, the following code gest triggered
    if event == '-LIST-' and len(values['-LIST-']):
        selected_value = get_selected_value(values['-LIST-'])
        choice_list = [choice for choice in face_choices if format_choice(choice) == selected_value]
        #print("choice_list is:")
        #print(choice_list)
        if (choice_list is None):
            print("Whoops, something went wrong when retrieving element from list")
        else:
            # Now we can get the original object back from the json file
            choice = choice_list[0]
            ####
            sg.popup('Get ready to smile %s!' % (format_choice(choice)))
            capture_and_save_images(choice)
            # After the line above completes, the loop will continue.

            
window.close()

# When everything done, release the capture
camera.release()

