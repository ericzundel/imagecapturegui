# imagecapturegui - A user interface example for a listbox of names

## Summary
This project is a UI for a
classroom project that captures images using OpenCV that will be used as
a database for a segment of machine learning.

## Details

- A workstation/Raspberry Pi will be setup at the entrance to the classroom. ![Setup at entrance to classroom](https://github.com/ericzundel/imagecapturegui/CameraAndButton.png)
- A sensor (or a push button) will trigger capturing images from the camera.
- As students will enter the classroom each day, they will choose their name from the drop down list
- Files will be written to a directory named 'images' with the
  first and last name concatenated together

### Database of student names

The database of student names is stored in JSON format in a file named 'face_chioces.json' in the same directory with the script.  Here is an example of the file:

```
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
```

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

Libraries to install are in requirements.txt.  You can install them with `pip -r requirements.txt`

After installing libraries, running `python imagacapture.py` should start the GUI
