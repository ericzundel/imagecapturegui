# See PySimpleGui/Demo_Listbox_Search_Filter.py
#
import PySimpleGUI as sg
import json

face_choices_file_path = 'face_choices.json'

def format_choice(choice_elem):
    """ Format one element of the JSON array for display.

    We'll just use the first_name and last_name fields in the json array
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
    layout = [[sg.Text('Listbox with search')],
          [sg.Input(size=(20, 1), enable_events=True, key='-INPUT-')],
          [sg.Listbox(list_values, size=(20, 4), enable_events=True, key='-LIST-')],
          [sg.Button('Chrome'), sg.Button('Exit')]]

    return sg.Window('Listbox with Search', layout)

def get_selected_value(value_list):
    """Retrieve the selected value as a scalar, not a one item list"""
    if (value_list is None):
        raise new Exception("Whoops, something went wrong in retrieving value from event")
    
    return value_list[0]

####################################################################
# Setup the User Interface
#

# Read the JSON file in
face_choices = read_face_choices()

# Format the names in the file for display in a listbox
names = [format_choice(elem) for elem in face_choices]
print("List of names found in JSON file is:", names)

# Create and display the main UI
window = build_window(names)

############################################################################
# UI Event Loop
#
# This loop executes until someone closes the main window.
#
while True:
    event, values = window.read()

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
            print("Found choice ", choice)
            ###
            ### *EDIT*
            ### In place of the next line you can call OpenCV 
            ### to capture from the camera
            ###
            sg.popup('Selected ', format_choice(choice))
            
window.close()
