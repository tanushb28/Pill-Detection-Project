'''
#Setup commands:
git clone https://github.com/THU-MIG/yolov10
cd yolov10
#if you're using venv
#replace pip with bin/pip
pip install -r requirements.txt
pip install -e .

#change MODEL_PATH to whereever your model is
#run (if venv then use bin/python)
python -i livestream_for_yolov10.py
#and then in python, run
>>> find_false_positive("example_filename.jpg", 0.7)
#at the end it prints a list of confidences of pills found
'''
MODEL_PATH='/storage/yolo-v10n-pills.pt'

from ultralytics import YOLOv10
import cv2

model = YOLOv10(MODEL_PATH)
#model = YOLOv10('./yolov10n.pt')

#min_conf is the minimum confidence of false positive in order to break and save the image
#	by default it's 50%, for the ones i found i used 70%
#save is the filename it'll save to, it'll also save to "fr" + save with the bounding boxes
#press q to quit
def find_false_positive(save="found.jpg", min_conf=0.5):
	cap = cv2.VideoCapture(0); 
	while True:
		ret, frame = cap.read()
		if not ret: break  
		
		results = model.predict([frame])
		frame = results[0].orig_img
		if (len(results[0].boxes) != 0 and max(results[0].boxes.conf) > min_conf):
			cv2.imwrite(save, frame)
		for box in results[0].boxes: 
			xB = int(box.xyxy[0][2])
			xA = int(box.xyxy[0][0])
			yB = int(box.xyxy[0][3])
			yA = int(box.xyxy[0][1])
			cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
		if (len(results[0].boxes) != 0 and max(results[0].boxes.conf) > min_conf):
			print("Found...")
			print(results[0].boxes.conf)
			cv2.imwrite("fr" + save, frame)
			break
		cv2.imshow("1", frame)
		if cv2.waitKey(1) == ord('q'): break
	cap.release()
	cv2.destroyAllWindows()

#should draw bounding boxes over detected objects
#there's a way to have nicer output with "annotation", i'll check that later
def webcam():
	cap = cv2.VideoCapture(0); 
	while True:
		ret, frame = cap.read()
		if not ret: break  
		
		results = model.predict([frame])
		frame = results[0].orig_img
		for box in results[0].boxes: 
			xB = int(box.xyxy[0][2])
			xA = int(box.xyxy[0][0])
			yB = int(box.xyxy[0][3])
			yA = int(box.xyxy[0][1])
			cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
		cv2.imshow("1", frame)
		if cv2.waitKey(1) == ord('q'): break
	cap.release()
	cv2.destroyAllWindows()

##uncomment so it runs automatically
#find_false_positive("example_filename.jpg", 0.7)
