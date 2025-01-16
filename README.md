# imagecapturegui - A user interface example for a listbox of names

## Summary
This project is a UI for a
classroom project that captures images using OpenCV that will be used as
a database for a segment of machine learning.

- imagecapture.py - A UI for capturing data from the Raspberry Pi camera and storing it
- facerecognition.py - a UI for trying to recognize an image using a pre-trained model
- hill-day-demo.py - A UI that was used for a demonstration in Washington DC that combines capture and recognition (but doesn't actually save data)
- backup_to_google_drive.py - A job run from cron to occasionally backup captured image data
- convert_model_to_tflite.py - A script to convert a trained model from keras to Tensorflow Lite format


## Details

- A workstation/Raspberry Pi will be setup at the entrance to the classroom. ![Setup at entrance to classroom](https://raw.githubusercontent.com/ericzundel/imagecapturegui/main/CameraAndButton.png)
- A sensor (or a push button) will trigger capturing images from the camera.
- As students will enter the classroom each day, they will choose their name from the drop down list
- Files will be written to a directory named 'images' with the
  first and last name concatenated together

### Database of student names

The database of student names is stored in JSON format in a file named 'face_chioces.json' in the same directory with the script.  Here is an example of the file:


    [
      {
	    "first_name" : "John",
	    "last_name" : "Doe"
      },
      {
	    "first_name" : "Sue",
	    "last_name" : "Smith"
      }
    ]

If you are not familiar with JSON format, it is used for data interchange,
usually over a network.

- '[]' indicates an array
- '{}' indicates a dictionary with "tag":"value" pairs separated by commas.

Note that the JSON interpreter is quite strict.

- There are no comment characters.
- Stray commas (like trailing commas) are not allowed.
- Tags and values in a dictionary must be delimted by double quotes.

Currently, only the tags "first_name" and "last_name" are used, but other values could be entered and used by the program.

## Ultrasonic sensor

The sensor is a widely available HC-SR04 ultrasonic sensor. It is wired up to pins on the Raspberry Pi
with the echo pin being wired with 2 resistors to create a voltage divider so we don't send 5V signals 
to the Raspberry Pi GPIO.

The sensor is controlled by a custom library that spawns a thread to read the sensor in a loop.
See proximity_sensor.py.  I'm not sure that the Raspberry Pi has real threading, but it seems to be 
OKish. In general, the sensor is not as responsive as I'd like so we substituted a pushbutton!

## Running the Code

Libraries to install are in requirements.txt.  You might be able to install them with `pip install -r requirements.txt`, but each version of Raspberry Pi OS seems to be different.

Raspberry Pi 4 with full image, I used:

    sudo apt install python3-opencv
    python -m venv venv
    venv/bin/pip install -r requirements.txt


After installing libraries, running `python imagecapture.py` should start the GUI

## Connecting to Google Drive

For backing up the data to google drive, you will need to save a file named 'credentials.json'
Go through the Goole Workspace and create an API key. See [Google Workspace](https://developers.google.com/workspace/guides/create-credentials)

## Running face recognition

Getting face recognition to work is a battle of python versioning.  I used Python 3.11 on my Windows machine and tensorflow-cpu version 2.15.0 to match the version of Tensorflow on Google Colab. Since there is little control over versioning, you'll likely run into issues if you try to run a model on another machine created with Google Colab.

I used the .tf format and it seemed to be compatible at the time, but not if you install Tensorflow 2.16.0!

## Notes on classroom use

To collect the data we used the script `facerecognition.py` To run the trained models we used `facerecognition4.py` which did a head to head comparison of 4 different student authored models.

## Notes on Hill Day demo 

Beth White and Dr Pascal Van Hentenryck took the hardware to Washington DC for Hill Day demonstration for US Congress and the NSF.  We reconfigured the hardware to have 2 hardware buttons - one to run the data collection and one to run student model. See`hill-day-demo.py`

### Pinout for Hill Day Demo

Physical Layout & Pin numbering: 

Pin 2 is at the top right when looking at the Pi from the rear. USB power connections ar on the bottom and USB connector is on the right.


   2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40
   1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39


Device Wiring based on physical layout:


    D1  X D4  X  X  X  X  X  X  X  X  X  X  X  X  X  X   X B2 B1
     X D2 D3  X  X  X  X  X  X  X  X  X  X  X  X  X  X   X  X BG 

    D1: Red wire for Display     : Pin 2 (5V)       : Display Power
    D2: Green wire for Display   : Pin 3 (SDA)      : Serial Port Data 
    D3: Yellow wire to Display   : Pin 5 (SCL)      : Serial Port Clock
    D4: Black wire to Display    : Pin 6 (GND) : Ground

    B1: Blue wire Button 1       : Pin 40 (GPIO 21) : Capture Data Button Sense
    B2: Purple wire  Button 2    : Pin 38 (GPIO 20) : Predict Button Sense
    BG: Black wire Button Ground : Pin 39 (GND)     : Capture Data Button Ground


### Troubleshooting

#### Backup scripts


There are two scripts used for backing up the image data. One is a full backup, the other is an incremental backup.

If your backups are not working, try running `backup_to_google_drive.py` or `incremental_backup_to_google_drive.py` from the commandline and look for an error.

If you get an error authenticating while running the backup scripts, you'll need to re-authenticat to google's developer API
```
google.auth.exceptions.RefreshError: ('invalid_grant: Bad Request', {'error': 'invalid_grant', 'error_description': 'Bad Request'})
```


- You'll need to be added to the list of valid developers in the Google APIs
- Bring up Chrome logged in to your organization's email account
- Remove the file `token.json` from the main directory where the script runs from
- Run the script again. You'll be prompted to use the web browser to authenticate.
- A new authentication token will be stored in `token.json`
```
