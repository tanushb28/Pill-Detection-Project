import cv2
import mediapipe as mp
import dlib
import numpy as np

# Init Mediapipe models for pose and hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Pose and hands setup with some confidence thresholds
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setting up Dlib models for face detection/recognition
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Background subtractor to detect movement
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Dict to keep track of known faces
faces = {}
current_id = 1

# Get face embeddings — basically, unique vectors for each face
def get_face_embedding(image, landmarks):
    return np.array(face_rec_model.compute_face_descriptor(image, landmarks, 1))

# Helper function for distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (Mediapipe works with RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    # Check for movement using background subtraction
    fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    movement_detected = np.sum(fg_mask) > 5000

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

        if gesture_detected:
            cv2.putText(frame, "Eating Gesture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Face detection and recognition part
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = detector(gray)

    for rect in detected_faces:
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        face_id = None

        # Get landmarks, compute face embeddings
        landmarks = predictor(gray, rect)
        face_embedding = get_face_embedding(frame, landmarks)

        # Check if we already know this face
        for id, saved_embedding in faces.items():
            dist = np.linalg.norm(face_embedding - saved_embedding)
            if dist < 0.6:
                face_id = id
                break

        # If it's a new face, assign a new ID
        if face_id is None:
            face_id = current_id
            faces[face_id] = face_embedding
            current_id += 1

        # Draw bounding box and label around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {face_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw all face landmarks for this person
        for i in range(68):
            point = landmarks.part(i)
            cv2.circle(frame, (point.x, point.y), 2, (255, 0, 0), -1)

    # Show the video stream
    cv2.imshow('Hand-to-Mouth Detection with Face Tracking', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
