""" Test the proximity sensor class"""

import proximity_sensor
import time

sensor = proximity_sensor.ProximitySensor(debug=True)

print("Main thread waiting")

while True:
    time.sleep(1)
    if sensor.is_triggered():
        print("Sensor was triggered.")
        print(
            "Distance is %f. Triggered Distance is %f"
            % (sensor.distance(), sensor.triggered_distance())
        )
