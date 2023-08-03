# Standard Library
import time
import datetime
import psutil
import sys
import os

# Machine Recognition
import tensorflow as tf
import cv2 as opencv2

# Graphing
import matplotlib.pyplot as graph
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Arduino
import serial

# top-level "constants"
ARDUINO_SERIAL_PORT = '/dev/ttyACM0'
ARDUINO_BAUD_RATE = 9600
TFLITE_MODEL_PATH = './plume_detection_model.tflite'
PICTURE_INTERVAL_SECONDS = 10
DATA_STORAGE_PATH = './webcam_live_data'

# non-constant test index, determined on start
TEST_INDEX = 0

if __name__ == '__main__':

    # 0 - Default USB Webcam
    print('[Init] Loading Camera')
    camera = opencv2.VideoCapture(0)

    # TFLite Model Analysis Machine
    print('[Init] Loading TFLite Model')
    model = ImageRecognitionModel(TFLITE_MODEL_PATH)

    # Current Frame
    frame_index = 0
    last_frame_time = time.localtime()

    # Gathered Data
    data = []

    # Establish Arduino Connection
    print('[Init] Establishing Serial Connection to Arduino')
    arduino_serial = serial.Serial(
        ARDUINO_SERIAL_PORT, 
        ARDUINO_BAUD_RATE, 
        timeout=1)
    arduino_serial.reset_input_buffer()

    # Ensure the data serialization path exists
    print('[Init] Generating Output Path')
    if not os.path.exists(DATA_STORAGE_PATH):
        os.makedirs(DATA_STORAGE_PATH)
        os.makedirs(DATA_STORAGE_PATH + '/frames')

    print('[Init] Calculating Test Index')
    latest_index = max([int(test.replace('test', '')) 
        for test in os.listdir(DATA_STORAGE_PATH + '/frames')])

    if latest_index == None:
        latest_index = 0

    # increment the test index from the previous one
    TEST_INDEX = latest_index + 1
    print('[Init] Test Index: ', TEST_INDEX)

    # Write to master test record (to begin the test)
    print('[Init] Generating/Appending Test to Master Log')
    master_list_file = open('./master_list.txt', 'a')
    master_list_file.write(f'test{TEST_INDEX}={str(datetime.now())}')
    master_list_file.close()

    # Loop until close
    while camera.isOpened():

        # Press 'Q' to close a window (if open)
        if opencv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_delta = time.localtime() - last_frame_time

        if time_delta.total_seconds() >= PICTURE_INTERVAL_SECONDS:
            frame_index += 1
            frame = take_picture(camera)
            frame_result = FrameData(model, arduino_serial, frame_index, frame)
            data.append(frame_result)
            display_frame(frame, frame_result)

        last_frame_time = time.localtime()

    print('[Exit] Releasing camera and ending camera stream')

    camera.release()
    opencv2.destroyAllWindows()

    print('[Data] Now serializing remaining data')

    save_and_analyze_data(data)

class FrameData:
    """
    Object representation of the full data garnered with every frame
    """
    def __init__(self, model, arduino_serial, frame_index, image):
        # timestamp and frame index
        self.timestamp = datetime.now()
        self.timestamp_str = str(timestamp)
        self.frame_index = frame_index

        # rather than storing the image, this will be serialized 
        # out of memory immediately (for performance)
        print('[Data] Serializing photo data to file')
        os.chdir(DATA_STORAGE_PATH + '/frames/test' + TEST_INDEX)
        self.image_path = f'frame-{self.timestamp_str}.jpg'
        opencv2.imwrite(self.image_path, image)

        # frame analysis
        print('[Model] Analyzing image..')
        self.analysis = model.run(image)
        self.classification = self.analysis[0][0]
        self.classification_confidence = self.analysis[0][1]
        print(f'[Model] Image result: {self.classification}, Confidence = {self.classificiation_confidence}')

        # environment/runtime health data collection
        self.temperature, self.humidity = generate_environment_data(arduino_serial)
        self.battery_use = generate_battery_usage_data()
        self.cpu_use = psutil.cpu_percent()
        self.memory_use = psutil.virtual_memory()[2]

    def to_string():
        """
        Returns a human-readable debug string for live viewing.
        Does NOT contain all of the data!
        """
        return ('frame: {}, class: {}, certainty: {}, ' + 
            'temp: {}, humidity: {}, cpu-use: {}, mem-use: {}').format(
            self.frame_index,
            self.classification,
            self.classification_confidence,
            self.humidity,
            self.temperature,
            self.cpu_use,
            self.memory_use)

    def to_csv():
        """
        Returns a CSV-serializable string of the data.
        Format requires it to be in the following order!
        """
        return ('{},{},{},{},{},{},{},{},{},{},{}').format(
            self.frame_index,
            self.timestamp_str,
            self.image_path,
            self.classification,
            self.classification_confidence,
            self.analysis,
            self.temperature,
            self.humidity,
            self.battery_use,
            self.cpu_use,
            self.memory_use)

# https://stackoverflow.com/questions/50443411/how-to-load-a-tflite-model-in-script
class ImageRecognitionModel:
    def __init__(self, model_path, labels, image_size=224):
        """
        Loads the TFLite image recognition/classifier model
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size = image_size

    def run(self, image):
        """
        Analyzes a given image/frame using the model loaded previously 

        args:
            image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """
        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output[0])

        # create list of ["label", probability], ordered descending probability
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        return sorted(label_to_probabilities, key=lambda element: element[1])

def take_picture(camera):
    complete, frame = camera.read()

    if not complete:
        sys.exit('Incomplete Camera Read')

    print(f'[Loop] Regular Interval Photo {frame_index} taken')
    return frame

def display_frame(frame, frame_result):
    """
    Visually displays the latest frame and its data post-analysis.
    """

    # applies the analysis result onto the frame to display
    opencv2.putText(frame, frame_result.to_string(), (7, 35), opencv2.FONT_HERSHEY_SIMPLEX, 
        1, (100, 255, 0), 3, opencv2.LINE_AA)

    # shows the frame
    opencv2.imshow('frame', frame)

def generate_environment_data(arduino_serial):
    """
    Connects to the Arduino to gather environment data from the
    DHT11 Temperature and Humidity Sensor
    """

    print('[Loop] Sending signal to generate environment data to Arduino')

    # sends the signal to generate the weather data
    arduino_serial.write(b"GEN_WEATHER_DATA")

    # recieves data (if sent correctly) -- [temperature, humidity]
    payload = arduino_serial.readline().decode('utf-8').rstrip()

    # converts the string array to ints
    temperature, humidity = [ eval(i) for i in payload.split(',') ]

    print('[Loop] Recieved environment data')

    return temperature, humidity

def generate_battery_usage_data():
    """
    Fetches the current battery usage at a given frame in KWh.

    NOT YET IMPLEMENTED!
    """
    return -1 # what a lazy cop out...

def save_and_analyze_data(data):
    """
    Calculates, saves, and displays generalizations from the experiment.
    It is intended for it to save the environment data (temperature and humdity), 
    successful plume recognitions, battery usage rates, and time stamps.
    """

    ### First part: data serialization

    print('[Data] Generating test data CSV file')
    os.chdir(DATA_STORAGE_PATH)
    output_file = open('./test' + TEST_INDEX + '.csv', 'a')

    # csv header
    output_file.write('frame_index,timestamp,' + 
        'image_path,classification,' + 
        'classification_confidence,' + 
        'analysis,temperature,' + 
        'humidity,battery_use,' + 
        'cpu_use,memory_use')

    temperature_list = []
    humidity_list = []
    classification_list = []
    battery_use_list = []
    cpu_use_list = []
    memory_use_list = []

    print('[Data] Serializing data to CSV')
    for entry in data:
        output_file.write(entry.to_csv())

        # unpack all of the statistics
        temperature_list.append(entry.temperature)
        humidity_list.append(entry.humidity)
        classification_list.append(entry.classification)
        battery_use_list.append(entry.battery_use)
        cpu_use_list.append(entry.cpu_use)
        memory_use_list.append(entry.memory_use)

    output_file.close()

    ### Second Part: Analysis

    print('[Data] Generating PDF analsis graphs file')
    with PdfPages(f'./test{TEST_INDEX}-analysis.pdf') as output_pdf:
        graph.rcParams['text.usetex'] = True
        graph.xlabel('frame_index')

        graph.figure()
        graph.plot(temperature_list)
        graph.title('temperature')
        graph.ylabel('temperature (Celsius)')
        output_pdf.savefig()
        graph.close()

        graph.figure()
        graph.plot(humidity_list)
        graph.title('humidity')
        graph.ylabel('humidity (g/kg)')
        output_pdf.savefig()
        graph.close()

        graph.figure()
        labels, counts = np.unique(a, return_counts=True)
        ticks = range(len(counts))
        graph.bar(ticks, counts, align='center')
        graph.xticks(ticks, labels)
        output_pdf.savefig()
        graph.close()

        graph.figure()
        graph.plot(battery_use_list)
        graph.title('battery usage')
        graph.ylabel('battery usage (KWh)')
        output_pdf.savefig()
        graph.close()

        graph.figure()
        graph.plot(cpu_use_list)
        graph.title('cpu usage')
        graph.ylabel('cpu usage (%)')
        output_pdf.savefig()
        graph.close()

        graph.figure()
        graph.plot(battery_use_list)
        graph.title('memory usage')
        graph.ylabel('memory usage (%)')
        output_pdf.savefig()
        graph.close()

        pdf_metadata = output_pdf.infodict()
        pdf_metadata['Title'] = f'test{TEST_INDEX} data graphs'
        pdf_metadata['CreationDate'] = str(datetime.now())
