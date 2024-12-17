#!/bin/python3
import cv2
from copy import copy
import mediapipe as mp
import dlib
import numpy as np
from ultralytics import YOLO

MODEL_PATH='/storage/mein/yolo-v10n-pills.pt'
model = YOLO(MODEL_PATH)

# Init Mediapipe models for pose and hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Pose and hands setup with some confidence thresholds
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dict to keep track of known faces
faces = {}
current_id = 1

pills = {}
pill_held_recently = -1
lips_touched_recently = -1
pill_is_being_swallowed = False

# Get face embeddings — basically, unique vectors for each face
def get_face_embedding(image, landmarks):
	return np.array(face_rec_model.compute_face_descriptor(image, landmarks, 1))

# Helper function for distance between two points
def calculate_distance(point1, point2):
	return np.linalg.norm(np.array(point1) - np.array(point2))

def area(xA, xB, yA, yB):
	return abs((xA - xB) * (yA - yB))

def pill_center(pill):
	return ((pill[0] + pill[2])/2, (pill[1] + pill[3])/2)

#computes a measure of overlap between boxes
def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def overlap(rect1, rect2):
	return iou(rect1, rect2) > 0.5

# Start video capture
cap = cv2.VideoCapture(0)

no_of_pills_swallowed = 0
frame_number = 0
while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	frame_number += 1

	# Convert frame to RGB (Mediapipe works with RGB)
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	height, width, _ = frame.shape

	lips_center = None
	gesture_detected = False

	# Get pose and hands landmarks
	pose_results = pose.process(frame_rgb)
	hands_results = hands.process(frame_rgb)

	# If pose is detected, get the lips coordinates
	if pose_results.pose_landmarks:
		landmarks = pose_results.pose_landmarks.landmark
		lips_center = [(landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].x + 
						landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].x) / 2,
					   (landmarks[mp_pose.PoseLandmark.MOUTH_LEFT].y + 
						landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT].y) / 2]
		lips_center = [int(lips_center[0] * width), int(lips_center[1] * height)]

	# Draw a circle at lips center if we got it
	if lips_center:
		cv2.circle(frame, tuple(lips_center), 5, (0, 255, 0), -1)

	#run only every third frame
	if frame_number % 3 == 0:
		results = model.predict([frame])
		frame = results[0].orig_img
		for box in results[0].boxes: 
			xB = int(box.xyxy[0][2])
			xA = int(box.xyxy[0][0])
			yB = int(box.xyxy[0][3])
			yA = int(box.xyxy[0][1])
			if (lips_center != None and xA < lips_center[0] and lips_center[0] < xB and yA < lips_center[1]):
				continue
			if (area(xA, xB, yA, yB) > 0.1 * height * width):
				continue
			print("Conf: ", max(results[0].boxes.conf))
			newpill = (xA, yA, xB, yB)
			newpill_is_new = True
			for pill in copy(pills):
				if overlap(pill, newpill): 
					pills.pop(pill)
					#the first value is how many frames back it was added into the list
					#the second value is whether it was being touched by the hand
					pills[newpill] = (0, False)
					newpill_is_new = False
				else:
					pills[pill] = (pills[pill][0] + 1, pills[pill][1])
			if newpill_is_new:
				pills[newpill] = (0, False)
				
	for pill in copy(pills):
		print(pill)
		if (pills[pill][0] > 5):
			if pills[pill][1] == True:
				print("PILL HOLDING MODE")
				cv2.putText(frame, "pill held", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
				pills.pop(pill)
				pill_held_recently = 0
			else:
				pills.pop(pill)
				pass
	print(pills)
	for pill in copy(pills):
		cv2.rectangle(frame, (pill[0], pill[1]), (pill[2], pill[3]), (0, 255, 0), 2)

	# Check for hands and gesture with index/thumb fingers
	if hands_results.multi_hand_landmarks:
		for hand_landmarks in hands_results.multi_hand_landmarks:
			index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
			thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

			# Draw the hand landmarks
			mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

			# Check if fingertips are near lips
			if lips_center:
				index_dist = calculate_distance(lips_center, [int(index_tip.x * width), int(index_tip.y * height)])
				thumb_dist = calculate_distance(lips_center, [int(thumb_tip.x * width), int(thumb_tip.y * height)])

				# If both fingers are close enough, gesture detected
				if index_dist < 50 and thumb_dist < 50:
					gesture_detected = True
					lips_touched_recently = 0
			for pill in pills:
				index_dist = calculate_distance(pill_center(pill), [int(index_tip.x * width), int(index_tip.y * height)])
				thumb_dist = calculate_distance(pill_center(pill), [int(thumb_tip.x * width), int(thumb_tip.y * height)])

				# If both fingers are close enough, gesture detected
				if index_dist < 50 and thumb_dist < 50:
					cv2.putText(frame, "pill picked up", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
					pill_touch_detected = True
					pills[pill] = (pills[pill][0], True)

		
		if gesture_detected:
			cv2.putText(frame, "Eating Gesture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	#these values were handpicked
	if lips_touched_recently > -1 and lips_touched_recently < 3 and pill_held_recently > -1 and pill_held_recently < 3:
		no_of_pills_swallowed += 1
		pill_is_being_swallowed = True
	if lips_touched_recently > -1 and lips_touched_recently < 5: 
		cv2.putText(frame, "Pill swallowed", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	if pill_held_recently != -1: pill_held_recently += 1
	if lips_touched_recently != -1: lips_touched_recently += 1

	# Show the video stream
	cv2.imshow('Hand-to-Mouth Detection with Face Tracking', frame)

	# Quit if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
