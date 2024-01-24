"""Code to read from the echo sensor and capture an image"""

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


##########################
# Setup the Ultrasonic sensor pins
GPIO.setmode(GPIO.BCM)

# Setup Trigger pin
GPIO.setup(TRIGGER_PIN, GPIO.OUT, initial=GPIO.LOW)
# Setup Echo pin
GPIO.setup(ECHO_PIN, GPIO.IN)

# Setup concurrency variables
SENSOR_LOCK = Lock()
SENSOR_TRIGGERED = False  # Protected by sensor_lock

def read_distance():
    distances = []
    num_over_threshold = 0

    # Make sure there are consecutive samples  that are below the threshold
    for i in range(NUM_SAMPLES):
        distance = read_one_sample()

        # Short circuit the loop
        if distance > DISTANCE_THRESHOLD:
            num_over_threshold = num_over_threshold + 1
        else:
            distances.append(distance)

        if num_over_threshold > MAX_OVER_THRESHOLD:
            return MAX_DISTANCE

    return sum(distances) / len(distances)


def read_one_sample():
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

def read_sensor_thread():
    global SENSOR_TRIGGERED

    while True:
        start_time = time.time()
        distance = read_distance()
        elapsed = start_time - time.time()
        if distance < MAX_DISTANCE:
            print("Distance: %02f. Elapsed time to read sensor: %f seconds" %
                  (distance, elapsed))
            with SENSOR_LOCK:
                SENSOR_TRIGGERED = True
            time.sleep(HYSTERESIS_SECS)
        time.sleep(0.01)


t = Thread(target=read_sensor_thread)
t.run()

while True:
    time.sleep(.5)
    with SENSOR_LOCK:
        if SENSOR_TRIGGERED:
            print("Main thread got trigger")
            SENSOR_TRIGGERED = False
