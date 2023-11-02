import cv2
import numpy as np
import os
import pickle
import sys

from helper_functions import resize_video

recognizer = "lbph"
training_data = "lbph_classifier.yml"  
threshold = 10e5             

max_width = 800          

def load_recognizer(option, training_data):
    if option == "eigenfaces":
        face_classifier = cv2.face.EigenFaceRecognizer_create()
    elif option == "fisherfaces":
        face_classifier = cv2.face.FisherFaceRecognizer_create()
    elif option == "lbph":
        face_classifier = cv2.face.LBPHFaceRecognizer_create()
    else:
        print("Os algoritmos disponíveis são: Eigenfaces, Fisherfaces e LBPH")
        sys.exit()

    face_classifier.read(training_data) #.yml
    return face_classifier

face_classifier = load_recognizer(recognizer, training_data)

face_names = {}
with open("face_names.pickle", "rb") as f:
    original_labels = pickle.load(f)

    face_names = {v: k for k, v in original_labels.items()}


def recognize_faces(network, face_classifier, orig_frame, face_names, threshold, conf_min=0.7):
    frame = orig_frame.copy()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            face_roi = gray[start_y:end_y,start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120)) 

            prediction, conf = face_classifier.predict(face_roi)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            pred_name = face_names[prediction] if conf <= threshold else "Not identified"

            text = "{} -> {:.4f}".format(pred_name, conf)
            cv2.putText(frame, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

cam = cv2.VideoCapture(0)

while(True):
    ret, frame = cam.read()

    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    processed_frame = recognize_faces(network, face_classifier, frame, face_names, threshold)

    cv2.imshow("Lendo a face do usuário", processed_frame)
    cv2.waitKey(1)

print ("Concluído!")
cam.release()
cv2.destroyAllWindows()