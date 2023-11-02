import cv2
import numpy as np
import os

from helper_functions import resize_video

detector = "ssd"  
max_width = 800           

def detect_face(face_detector, orig_frame):
    frame = orig_frame.copy()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return frame

def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy()  
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            if (start_x<0 or start_y<0 or end_x > w or end_y > h):

                continue

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            if show_conf:
                text_conf = "{:.2f}%".format(confidence * 100)
                cv2.putText(frame, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

if detector == "haarcascade":

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
else:

    network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()


    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    if detector == "haarcascade":
        processed_frame = detect_face(face_detector, frame)
    else:  
        processed_frame = detect_face_ssd(network, frame)

    cv2.imshow("Lendo a face do usuário", processed_frame)
    cv2.waitKey(1)

print ("Concluído!")
cam.release()
cv2.destroyAllWindows()