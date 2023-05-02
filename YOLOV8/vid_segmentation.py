import numpy as np
import cv2
from ultralytics import YOLO
import random

# opening the file in read mode
my_file = open("class.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    detection_colors.append((b,g,r))

# load a pretrained YOLOv8n model
model = YOLO("model/best.pt", "v8") 

# Vals to resize video frames | small frame optimise the run 
# im resizing based on dataset image sizes
frame_wid = 640
frame_hyt = 640

# cap = cv2.VideoCapture(1) #for camera
cap = cv2.VideoCapture("greenhouse_test.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run 
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.70, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)
            # Visualize the results on the frame
            annotated_frame = detect_params[0].plot()

    else:
        # Break the loop if the end of the video is reached
        break

    # Display the resulting frame
    cv2.imshow('Greenhouse Plants', annotated_frame)

    # Terminate run when "Q" pressed
    #does not work on vscode terminal, needs to use keyboard interrupt of ctrl + c
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()