# USAGE
# python pi_detect_drowsinessv2.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat
# python pi_detect_drowsiness.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1

#sudo arecord -D plughw:1,0 test.wav
#export GOOGLE_APPLICATION_CREDENTIALS=/home/pi/Raehoon.json


# import the necessary packages
from __future__ import division
from imutils.video import VideoStream
from imutils import face_utils
import RPi.GPIO as GPIO
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import serial
import playsound
import pygame
import re
import sys
import random


port="/dev/ttyACM0"

serialFromArduino = serial.Serial(port,9600)
serialFromArduino.flushInput()




from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 44100
CHUNK = int((44100/16000)*1024)#int(RATE / 10)  # 100ms
global COUNTER
COUNTER=0
flag=1
arr=['바나나','할머니','배철수','몰라','다시']



class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()


    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                   
                   
                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))
        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()
            test=transcript.split(' ')
        
            print(test)
        
            for i in arr:
                
             
                    
                if i in test and i != '다시':
                    print("here")
                    
                    pygame.mixer.init()
                    pygame.mixer.music.load('/home/pi/dingdongdang.mp3')
                    pygame.mixer.music.play()
                    time.sleep(2)            
                    pygame.mixer.music.load('/home/pi/daddy1.mp3')
                    pygame.mixer.music.play()
                    COUNTER=0
                    return 0
                if i == '다시' and i in test:
                    pygame.mixer.init()
                    string =f'/home/pi/quiz{b}.mp3'
                    pygame.mixer.music.load(string)
                    pygame.mixer.music.play()
                    num_chars_printed =0
                    
                else:
                    num_chars_printed = len(transcript)

                
                    
            
        else:
            print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0




def euclidean_dist(ptA, ptB):
    # compute and return the euclidean distance between the two
    # points
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = euclidean_dist(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0,
    help="boolean used to indicate if TraffHat should be used")
args = vars(ap.parse_args())

# check to see if we are using GPIO/TrafficHat as an alarm
if args["alarm"] > 0:
    from gpiozero import TrafficHat
    th = TrafficHat()
    print("[INFO] using TrafficHat alarm...")
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off

ALARM_ON = False

# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

#stt
language_code = 'ko-KR'  # a BCP-47 language tag

client = speech.SpeechClient()
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code=language_code)
streaming_config = types.StreamingRecognitionConfig(
    config=config,
    interim_results=True)#, single_utterance=True)

# ultrasonic_sensor

GPIO.setmode(GPIO.BCM)

trig = 2
echo = 3
GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)

COUNTTMP=0
i=0
global b

# loop over frames from the video stream
try:
        while True:
                # grab the frame from the threaded video file stream, resize
                # it, and convert it to grayscale
                # channels)
                i=i+1
                COUNTTMP=0
                frame = vs.read()
                frame = imutils.resize(frame, width=500 )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale frame
                rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
                        minNeighbors=5, minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE)
                
   

                # loop over the face detections
                for (x, y, w, h) in rects:
                        # construct a dlib rectangle object from the Haar cascade
                        # bounding box
                        rect = dlib.rectangle(int(x), int(y), int(x + w),
                                int(y + h))

                        # determine the facial landmarks for the face region, then
                        # convert the facial landmark (x, y)-coordinates to a NumPy
                        # array
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)

                        # extract the left and right eye coordinates, then use the
                        # coordinates to compute the eye aspect ratio for both eyes
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)

                        # average the eye aspect ratio together for both eyes
                        ear = (leftEAR + rightEAR) / 2.0

                        # compute the convex hull for the left and right eye, then
                        # visualize each of the eyes
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                        # check to see if the eye aspect ratio is below the blink
                        # threshold, and if so, increment the blink frame counter
                       
                        
                        
                        if ear < EYE_AR_THRESH:
                                
                                COUNTER += 1
                                #ultrasonic
                                GPIO.output(trig, False)
                                time.sleep(0.1)
                                GPIO.output(trig, True)
                                time.sleep(0.0001)
                                GPIO.output(trig, False)
                                while GPIO.input(echo)==0:
                                        pulse_start = time.time()
                                while GPIO.input(echo)==1:
                                        pulse_end = time.time()
                                pulse_duration = pulse_end-pulse_start
                                
                                distance = pulse_duration*17000
                                distance = round(distance, 2)
                                # if the eyes were closed for a sufficient number of
                                # frames, then sound the alarm
                               
                                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                    cv2.putText(frame, "DROWSINESS ALERT!", (200, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    pygame.mixer.init()
                                    pygame.mixer.music.load('/home/pi/alarm.wav')
                                    pygame.mixer.music.play()
                                    time.sleep(5)
                                    # if the alarm is not on, turn it on
                                    # flag1=1
                                    if distance>8 :
                                            # draw an alarm on the frame
                                            pygame.mixer.init()
                                            pygame.mixer.music.load('/home/pi/head1.mp3')
                                            pygame.mixer.music.play()
                                            #print(f"Distance is too far : {distance}")
                                            print(f"Distance is too far : 3800") 
                                            time.sleep(5)
                               
                                    for i in range(10):
                                        input_s=serialFromArduino.readline()
                                        COUNTTMP+=int(input_s)
                                   
                                    
                                    
                                    COUNTTMP=COUNTTMP//5
                                 
                                    
                                    if COUNTTMP>300 and flag ==0:
                                        
                                            pygame.mixer.init()
                                            pygame.mixer.music.load('/home/pi/window1.mp3')
                                            pygame.mixer.music.play()
                                            print(f"Co2 ppm is too high : {COUNTTMP}") 
                                            time.sleep(10)

                                    
                                    a=random.randint(2,2)
                                    x=0
                                    if flag==0:
                                    
                                        b=random.randint(1,3)
                                        string =f'/home/pi/quiz{b}.mp3'
                                        print(f"Play Quiz{b}")
                                        pygame.mixer.init()
                                        pygame.mixer.music.load(string)
                                        pygame.mixer.music.play()
                                        #time.sleep(3)
                                        with MicrophoneStream(RATE, CHUNK) as stream:
                                            audio_generator = stream.generator()
                                            requests = (types.StreamingRecognizeRequest(audio_content=content)
                                                        for content in audio_generator)
                                            responses = client.streaming_recognize(streaming_config, requests)
                                            
                                            # Now, put the transcription responses to use.
                                            listen_print_loop(responses)
                                            COUNTER=0
                                            flag==1
                
                                    elif flag ==1:                                        
                                        pygame.mixer.init()
                                        pygame.mixer.music.load('/home/pi/handle1.mp3')
                                        pygame.mixer.music.play()
                                        time.sleep(7)
                                        while True:
                                                input_s=serialFromArduino.readline()
                                                input=int(input_s)
                                      
                                                x+=1           
                                                if input>10 and input<50 :   
                                                        pygame.mixer.init()
                                                        pygame.mixer.music.load('/home/pi/sleepy.mp3')
                                                        pygame.mixer.music.play()
                                                        time.sleep(6)            
#                                                         pygame.mixer.init()
#                                                         pygame.mixer.music.load('/home/pi/daddy1.mp3')
#                                                         pygame.mixer.music.play()
#                                                         time.sleep(20)
                                                        COUNTER=0
                                                        flag=0
                                                        
                                                        break
                                                elif x>=30:
                                                        pygame.mixer.init()
                                                        print(f"Input Pressure more than 5! Pressure: {input}")
                                                        pygame.mixer.music.load('/home/pi/handle1.mp3')
                                                        pygame.mixer.music.play()
                                                        x=0
                                    else:
                                         flag=0
                                                    
                                                        #time.sleep(7)
                                           
                        # otherwise, the eye aspect ratio is not below the blink
                        # threshold, so reset the counter and alarm
                        else:
                                COUNTER = 0
                                
                        # draw the computed eye aspect ratio on the frame to help
                        # with debugging and setting the correct eye aspect ratio
                        # thresholds and frame counters
                        cv2.putText(frame, "Eye Aspect Ratio: {:.3f}".format(ear), (200, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        

                          
                # show the frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
         
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                        break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
except:
        GPIO.cleanup()
