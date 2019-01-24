from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import winsound
import argparse
import imutils
import time
import dlib
import cv2

# To be used later for playing sound in windows systems
#winsound.PlaySound("audio_1.wav",  winsound.SND_ALIAS)

def alarm_sound():
	winsound.PlaySound("alarm.wav", winsound.SND_ALIAS)

def EAR(eye):
	A = dist.euclidean(eye[1], eye[5]) #distnace between P1(x,y) and P5(x,y)
	B = dist.euclidean(eye[2], eye[4]) #distnace between P2(x,y) and P4(x,y)
	C = dist.euclidean(eye[0], eye[3]) #distnace between P0(x,y) and P3(x,y)

#   eye_ascpect_ratio = dist(P1,P5)+dist(P2,P4)/2*dist(P0,P3)

	EAR = (A+B)/C

	return EAR

# Following command line argument needs to be parsed:
#  <name_of_this_python_file>.py  -p <path to facial landmark detector> -a <path to .wav sound file for alarm> -w <index of webcam of the system(0 is default, rest 1,2,3...) > 

ap = argparse.ArgumentParser()

# 'shape-predictor' is the path where facial landmark detection of dlib, link to download: 
#ap.add_argument("-p", "--shape-predictor", required=True,
#	help="path to facial landmark predictor")
# 'alarm' is the path of audio file to be played as alarm
#ap.add_argument("-a", "--alarm", type=str, default="",
#	help="path alarm .WAV file")
#'webcam' is the index of the webcam to be used
#ap.add_argument("-w", "--webcam", type=int, default=0,
#	help="index of webcam on system")

#defining the threshold value for EAR
EAR_thresh = 0.50
#defining no of frames for which the EAR should be below EAR_thresh
EAR_no_of_frames = 30

#defining counter value, where counter exceeding EAR_no_of_frames turns on the alarm
COUNTER = 0
#defining alarm state 
Alarm_on = False

# initialize dlib's face detector which uses Histogram Of Oriented Gradient and the facial landmark detector discussed previously

#print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grabing the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#--------Now we start the actual programming------------------------------

#starting the video stream
#print("[INFO] starting video stream thread...")
#webcam is provided in the command line as described earlier
vs  = VideoStream(0).start()
#to warm up sensor, we pause
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = EAR(leftEye)
		rightEAR = EAR(rightEye)
 
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
		if ear < EAR_thresh:
			COUNTER += 1
 
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER >= EAR_no_of_frames :
				# if the alarm is not on, turn it on
				if not ALARM_ON:
					ALARM_ON = True

 
					# check to see if an alarm file was supplied,
					# and if so, start a thread to have the alarm
					# sound played in the background
					
					Thread(target = alarm_sound).start() 


				  
 
				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			COUNTER = 0
			ALARM_ON = False




		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
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

