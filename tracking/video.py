#!/bin/python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import code #to break into python prompt

face_model_path = '/storage/downloads/face_landmarker.task'
hand_model_path = '/storage/downloads/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

VisionRunningMode = mp.tasks.vision.RunningMode

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw_face_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=VisionRunningMode.VIDEO)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    num_hands=2,
    min_tracking_confidence=0.3,
    min_hand_presence_confidence=0.1,
    running_mode=VisionRunningMode.VIDEO)

def dump_into_csv(landmark_results):
    landmarks = [(l.x, l.y, l.z) for l in landmark_results[0]]
    with open("out.txt", 'w') as outfile:
        for (x, y, z) in landmarks:
            print(f"{x}, {y}, {z},", file=outfile) 


#-----------------------------

import cv2
import numpy as np
from time import perf_counter, perf_counter_ns, sleep
import sys

print("Opening video file ", sys.argv[1])
cap = cv2.VideoCapture(sys.argv[1])
width     = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height    = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_fps = cap.get(cv2.CAP_PROP_FPS)
print("video dimensions: ", width, height)
print("video fps: ", video_fps)

if not cap.isOpened():
    print("Cannot open video file")
    exit()

frame_number = 0
with FaceLandmarker.create_from_options(face_options) as face_landmarker:
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        ret, frame = cap.read()
        timestamp = perf_counter_ns()
        last_timestamp = timestamp
        while True:
        #camera output by default is in BGR format, 
        #if you forget to convert the model sees odd colors and fails silently
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            video_timestamp = frame_number * (1/video_fps) * 1000 
            face_result = face_landmarker.detect_for_video(mp_image, int(video_timestamp))
            hand_result = hand_landmarker.detect_for_video(mp_image, int(video_timestamp))
            ret, frame = cap.read()
            #frame = cv2.flip(frame, 1)
            timestamp = perf_counter_ns()
            framerate = 1000*1000*1000/(timestamp - last_timestamp)
            last_timestamp = timestamp
            print("Framerate: ", framerate)

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            output_image = frame
            if (len(face_result.face_landmarks) > 0):
                #fl = [(int(width * l.x), int(height * l.y)) for l in face_result.face_landmarks[0]]
                face_landmark_i = face_result.face_landmarks[0][frame_number % 481]
                coords = (face_landmark_i.x * width, face_landmark_i.y * height)
                coords = (int(coords[0]), int(coords[1]))
                output_image = cv2.circle(output_image, coords, radius=10,color=(0,255,0),thickness=-1)
            if (len(hand_result.hand_landmarks) > 0):
                #hl = [(int(width * l.x), int(height * l.y)) for l in hand_result.hand_landmarks[0]]
                hand_landmark_i = hand_result.hand_landmarks[0][frame_number % 21]
                coords = (hand_landmark_i.x * width, hand_landmark_i.y * height)
                coords = (int(coords[0]), int(coords[1]))
                output_image = cv2.circle(output_image, coords, radius=10,color=(255,255,0),thickness=-1)
            print((frame_number % 481, frame_number % 21))
#COMMENT OUT NEXT TWO LINES TO STOP DRAWING ALL LANDMARKS
            output_image = draw_face_landmarks_on_image(output_image, face_result)
            output_image = draw_hand_landmarks_on_image(output_image, hand_result)

            cv2.imshow('frame', output_image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_face = not show_face
            elif key == ord('h'):
                show_hand = not show_hand
            #code.interact(local=locals())
            frame_number+=1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
