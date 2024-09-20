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
    cv.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

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

face_done = False
hand_done = False
def face_callback(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global face_result, face_done
    face_result = result
    face_done = True

def hand_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global hand_result, hand_done
    hand_result = result
    hand_done = True

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=face_model_path),
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=face_callback)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_model_path),
    num_hands=2,
    min_tracking_confidence=0.1,
    min_hand_presence_confidence=0.1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=hand_callback)

face_landmarker = FaceLandmarker.create_from_options(face_options)

def dump_into_csv(landmark_results):
    landmarks = [(l.x, l.y, l.z) for l in landmark_results[0]]
    with open("out.txt", 'w') as outfile:
        for (x, y, z) in landmarks:
            print(f"{x}, {y}, {z},", file=outfile) 


#-----------------------------

import cv2 as cv
import numpy as np
from time import perf_counter, perf_counter_ns, sleep

cap = cv.VideoCapture(0)
#width, height = 1920, 1080
#cap.set(cv.CAP_PROP_FRAME_WIDTH, 224)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT, 224)
print(cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

with FaceLandmarker.create_from_options(face_options) as face_landmarker:
    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        ret, frame = cap.read()
        timestamp = perf_counter_ns()
        last_timestamp = timestamp
        face_done, hand_done = True, True
        face_result, hand_result = None, None
        show_face, show_hand = True, True
        while True:
	    #camera output by default is in BGR format, 
	    #if you forget to convert the model sees odd colors and fails silently
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            if face_done and hand_done:
                face_done, hand_done = False, False
                face_landmarker.detect_async(mp_image, timestamp // 1000 // 1000)
                hand_landmarker.detect_async(mp_image, timestamp // 1000 // 1000)
            ret, frame = cap.read()
            frame = cv.flip(frame, 1)
            timestamp = perf_counter_ns()
            framerate = 1000*1000*1000/(timestamp - last_timestamp)
            last_timestamp = timestamp
            print("Framerate: ", framerate)

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            output_image = frame
            if face_result and hand_result:
                if show_face: output_image = draw_face_landmarks_on_image(output_image, face_result)
                if show_hand: output_image = draw_hand_landmarks_on_image(output_image, hand_result)

            cv.imshow('frame', output_image)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('f'):
                show_face = not show_face
            elif key == ord('h'):
                show_hand = not show_hand
            #code.interact(local=locals())
            sleep(0.1)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
