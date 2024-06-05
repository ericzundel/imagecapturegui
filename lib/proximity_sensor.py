""" Module to use an Ultrasonic sensor to detect when someone is walking by.

Intended to be used with an Ultrasonic sensor attached to a 
Raspberry Pi on a GPIO port.

NB(ericzundel) I made this kind of complicated to try and get it to 
reliably detect a person. It worked, but it wasn't reliable enough for 
us to use.
"""
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
DEFAULT_ECHO_PIN = 17
DEFAULT_TRIGGER_PIN = 4

# SENSE_WAIT_TIME: Controls how frequently the sensor is checked.
# With a value of .05 this task takes 25% of the CPU on a RPI3+
# With a value of .1 it takes as little as 12% of the CPU on a RPI3+
SENSE_WAIT_TIME = 0.1  # seconds between checking the sensor

ECHO_TIMEOUT = 0.25  # Wait at most 250 MS for echo response
DEFAULT_DISTANCE_THRESHOLD = 100  # Detect objects closer than 100cm / 1m
NUM_SAMPLES = 4  # Check the sensor 4 times for consecutive readings
CERTAINTY_THRESHOLD = .75  # % samples positive to report a positive result.
MAX_DISTANCE = 1000  # Sentinel value for something not detected
HYSTERESIS_SECS = .25  # Time to wait after a positive distance result

# Computed constant from variables above
MAX_OVER_THRESHOLD = int(NUM_SAMPLES - (NUM_SAMPLES * CERTAINTY_THRESHOLD))


class ProximitySensor:
    """Class that uses a HC-SR04 sensor on a Raspberry Pi to detect"""

    def __init__(self, echo_pin=DEFAULT_ECHO_PIN,
                 trigger_pin=DEFAULT_TRIGGER_PIN,
                 distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
                 debug=False):
        """Start reading from the HC-SR04 sensor in the background"""

        ##########################
        # Setup the Ultrasonic sensor pins
        GPIO.setmode(GPIO.BCM)

        self._trigger_pin = trigger_pin
        self._echo_pin = echo_pin
        self._distance_threshold = distance_threshold
        self._debug = debug

        # Setup Trigger pin
        GPIO.setup(self._trigger_pin, GPIO.OUT, initial=GPIO.LOW)
        # Setup Echo pin
        GPIO.setup(self._echo_pin, GPIO.IN)

        # Setup concurrency variables
        self._sensor_lock = Lock()
        self._sensor_triggered = False  # Protected by sensor_lock
        self._distance = MAX_DISTANCE   # Protected by sensor_lock
        self._triggered_distance = 0    # Protected by sensor_lock
        self._stop_now = False          # Protected by sensor_lock

        if self._debug:
            print("ProximitySensor: Trigger pin is %d" % (self._trigger_pin))
            print("ProximitySensor: Echo pin is %d" % (self._echo_pin))
            print("ProximitySensor: Distance Threshold is %d" %
                  (self._distance_threshold))
            print("ProximitySensor: Max over threshold is %d" %
                  (MAX_OVER_THRESHOLD))
            print("ProximitySensor: Max Distance %d" %
                  (MAX_DISTANCE), flush=True)

        # Start a background thread
        self._thread = Thread(target=self._read_sensor_thread, args=[])
        self._thread.start()

    def _read_distance(self):
        distances = []
        num_over_threshold = 0

        # Make sure there are consecutive samples  that are below the threshold
        for i in range(NUM_SAMPLES):
            distance = self._read_one_sample()

            # Short circuit the loop
            if distance > self._distance_threshold:
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
        GPIO.output(self._trigger_pin, GPIO.HIGH)  # Set trig high
        time.sleep(0.00001)  # 10 micro seconds 10/1000/1000
        GPIO.output(self._trigger_pin, GPIO.LOW)  # Set trig low
        pulselen = None
        timestamp = time.monotonic()

        while GPIO.input(self._echo_pin) == GPIO.LOW:
            if time.monotonic() - timestamp > ECHO_TIMEOUT:
                raise TimeoutError("Timed out")
            timestamp = time.monotonic()

        # track how long pin is high
        while GPIO.input(self._echo_pin) == GPIO.HIGH:
            if time.monotonic() - timestamp > ECHO_TIMEOUT:
                raise TimeoutError("Timed out")
        pulselen = time.monotonic() - timestamp
        pulselen *= 1000000  # convert to us to match pulseio
        if pulselen >= 65535:
            raise TimeoutError("Timed out")
        # positive pulse time, in seconds, times 340 meters/sec, then
        # divided by 2 gives meters. Multiply by 100 for cm
        # 1/1000000 s/us * 340 m/s * 100 cm/m * 2 = 0.017
        return pulselen * 0.017

    def _read_sensor_thread(self):
        global SENSOR_TRIGGERED

        if (self._debug):
            print("ProximitySensor: Thread started", flush=True)

        while True:
            start_time = time.time()

            # Check to see if the thread should exit
            with self._sensor_lock:
                if self._stop_now:
                    return

            distance = MAX_DISTANCE
            try:
                distance = self._read_distance()
            except TimeoutError:
                pass

            with self._sensor_lock:
                self._distance = distance

            elapsed = time.time() - start_time
            if distance < MAX_DISTANCE:
                if self._debug:
                    print(" ProximitySensor: Distance: %02f. Elapsed time to read sensor: %f seconds" %
                          (distance, elapsed), flush=True)
                with self._sensor_lock:
                    self._sensor_triggered = True
                    self._triggered_distance = distance
                time.sleep(HYSTERESIS_SECS)
            time.sleep(SENSE_WAIT_TIME)

    def is_triggered(self):
        """Returns True if the sensor triggered since this method was called"""
        result = False
        with self._sensor_lock:
            if self._sensor_triggered:
                result = True
                self._sensor_triggered = False
        return result

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

    def deinit(self):
        with self._sensor_lock:
            self._stop_now = True
        self._thread.join()

    class TimeoutError(RuntimeError):
        def __init__(self, message):
            super().__init__(self, message)
