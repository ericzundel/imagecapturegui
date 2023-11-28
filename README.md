# imagecapturegui - A user interface example for a listbox of names

## Summary
This project is intended to be a starting point to develop a UI for a
classroom project that captures images using OpenCV that will be used as
a database for a segment of machine learning.

## Details

- A workstation will be setup at the entrance to the classroom.
- As students will enter the classroom each day, they will choose their name from the drop down list
- The program will then capture a series of images of their face.
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

## Running the code

Libraries to install are in requirements.txt.  You can install them with `pip -r requirements.txt`
