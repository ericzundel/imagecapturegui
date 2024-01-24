"""Test code to read from the echo sensor and capture an image"""

from threading import Thread, Lock
import time

try:
    import RPi.GPIO as GPIO
except RuntimeError:
    print(
        "Error importing RPi.GPIO!  ",
        "This is probably because you need superuser privileges.  ",
        "You can achieve this by using 'sudo' to run your script",
    )
ECHO_PIN = 17
TRIGGER_PIN = 4
ECHO_TIMEOUT = 0.25  # Wait at most 250 MS for echo response
DISTANCE_THRESHOLD = 100  # Detect objects closer than 100cm / 1m
NUM_SAMPLES = 4  # Check the sensor 4 times for consecutive readings
CERTAINTY_THRESHOLD = .75  # % samples positive to report a positive result.
MAX_DISTANCE = 1000  # Sentinel value for something not detected
HYSTERESIS_SECS = .25  # Time to wait after a positive distance result

# Computed constant from variables above
MAX_OVER_THRESHOLD = int(NUM_SAMPLES - (NUM_SAMPLES * CERTAINTY_THRESHOLD))
print("Max over threshold is %d" % (MAX_OVER_THRESHOLD))

class ProximitySensor:
    """Class that uses a HC-SR04 sensor on a Raspberry Pi to detect"""
    def __init__(self):
        """Start reading from the HC-SR04 sensor in the background"""

        ##########################
        # Setup the Ultrasonic sensor pins
        GPIO.setmode(GPIO.BCM)

        # Setup Trigger pin
        GPIO.setup(TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)
        # Setup Echo pin
        GPIO.setup(ECHO_PIN, GPIO.IN)

        # Setup concurrency variables
        self._sensor_lock = Lock()
        self._sensor_triggered = False  # Protected by sensor_lock
        self._distance = MAX_DISTANCE   # Protected by sensor_lock
        self._triggered_distance = 0    # Protected by sensor_lock

        # Start a background thread
        self._thread = Thread(target=self._read_sensor_thread, args=[self])
        self._thread.start()

    def _read_distance(self):
        distances = []
        num_over_threshold = 0

        # Make sure there are consecutive samples  that are below the threshold
        for i in range(NUM_SAMPLES):
            distance = self._read_one_sample()

            # Short circuit the loop
            if distance > DISTANCE_THRESHOLD:
                num_over_threshold = num_over_threshold + 1
            else:
                distances.append(distance)

            if num_over_threshold > MAX_OVER_THRESHOLD:
                return MAX_DISTANCE

        return sum(distances) / len(distances)

    def _read_one_sample(self):
        """Blocks while reading from the Ultrasonic sensor.

        Usually takes about 20ms
        """
        GPIO.output(TRIGGER_PIN, GPIO.HIGH)  # Set trig high
        time.sleep(0.00001)  # 10 micro seconds 10/1000/1000
        GPIO.output(TRIGGER_PIN, GPIO.LOW)  # Set trig low
        pulselen = None
        timestamp = time.monotonic()

        while GPIO.input(ECHO_PIN) == GPIO.LOW:
            if time.monotonic() - timestamp > ECHO_TIMEOUT:
                raise RuntimeError("Timed out")
            timestamp = time.monotonic()

        # track how long pin is high
        while GPIO.input(ECHO_PIN) == GPIO.HIGH:
            if time.monotonic() - timestamp > ECHO_TIMEOUT:
                raise RuntimeError("Timed out")
        pulselen = time.monotonic() - timestamp
        pulselen *= 1000000  # convert to us to match pulseio
        if pulselen >= 65535:
            raise RuntimeError("Timed out")
        # positive pulse time, in seconds, times 340 meters/sec, then
        # divided by 2 gives meters. Multiply by 100 for cm
        # 1/1000000 s/us * 340 m/s * 100 cm/m * 2 = 0.017
        return pulselen * 0.017

    def _read_sensor_thread(self):
        global SENSOR_TRIGGERED

        while True:
            start_time = time.time()
            distance = self._read_distance()
            with self._sensor_lock:
                self._distance = distance
            elapsed = start_time - time.time()
            if distance < MAX_DISTANCE:
                print("Distance: %02f. Elapsed time to read sensor: %f seconds" %
                      (distance, elapsed))
                with self._sensor_lock:
                    SENSOR_TRIGGERED = True
                    self._triggered_distance = distance
                time.sleep(HYSTERESIS_SECS)
            time.sleep(0.01)

    def is_triggered(self):
        """Returns True if the sensor triggered since this method was called"""
        with self._sensor_lock:
            if self._sensor_triggered:
                print("Main thread got trigger")
                self._sensor_triggered = False

    def distance(self):
        """Returns the latest distance from the last reading of the sensor.

        Note: this reading is the latest and may not correspond to the distance
        computed when the sensor was triggered. See triggered_distance()
        """
        result = 0
        with self._sensor_lock:
            result = self._distance
        return result

    def triggered_distance(self):
        """Returns the  distance from the last reading of the sensor when triggered.

        Note: this reading is NOT the latest reading. See distance()
        """
        result = 0
        with self._sensor_lock:
            result = self._triggered_distance
        return result
