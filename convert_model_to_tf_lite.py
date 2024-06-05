import tensorflow as tf
import os

saved_model_dir = "./2024model/"

model_name = "student_recognition_2024_32bit"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(
    "%s/%s.tf" % (saved_model_dir, model_name))  # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('%s/%s.tflite' % (saved_model_dir, model_name), 'wb') as f:
    f.write(tflite_model)
