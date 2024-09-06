#Setup commands:
git clone https://github.com/THU-MIG/yolov10
cd yolov10
#if you're using venv
#replace pip with bin/pip
pip install -r requirements.txt
pip install -e .

cd ..
#in the python script, change MODEL_PATH to whereever your model is
#run (if venv then use bin/python)
python -i livestream_for_yolov10.py
#and then in python run
#>>> find_false_positive("example_filename.jpg", 0.7)
#at the end it prints a list of confidences of pills found
