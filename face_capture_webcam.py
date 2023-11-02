import cv2
import numpy as np
import os
import re

from helper_functions import resize_video

detector = "ssd"  
max_width = 800          

max_samples = 20   
starting_sample_number = 0  

def parse_name(name):
    name = re.sub(r"[^\w\s]", '', name) 
    name = re.sub(r"\s+", '_', name)   
    return name

def create_folders(final_path, final_path_full):
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    if not os.path.exists(final_path_full):
        os.makedirs(final_path_full)

def detect_face(face_detector, orig_frame):
    frame = orig_frame.copy()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.1, 5)

    face_roi = None

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face_roi = orig_frame[y:y + h, x:x + w]  
        face_roi = cv2.resize(face_roi, (140, 140)) 

    return face_roi, frame

def detect_face_ssd(network, orig_frame, show_conf=True, conf_min=0.7):
    frame = orig_frame.copy() 
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    detections = network.forward()

    face_roi = None
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_min:
            bbox = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = bbox.astype("int")

            if (start_x<0 or start_y<0 or end_x > w or end_y > h):

                continue

            face_roi = orig_frame[start_y:end_y,start_x:end_x]
            face_roi = cv2.resize(face_roi, (90, 120))
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)  
            if show_conf:
                text_conf = "{:.2f}%".format(confidence * 100)
                cv2.putText(frame, text_conf, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return face_roi, frame

if detector == "ssd":

    network = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
else:

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cam = cv2.VideoCapture(0)

folder_faces = "dataset/"      
folder_full = "dataset_full/"  

person_name = input('Coloque seu nome: ')
person_name = parse_name(person_name)

final_path = os.path.sep.join([folder_faces, person_name])
final_path_full = os.path.sep.join([folder_full, person_name])
print("Todas as fotos serão salvas em {}".format(final_path))


create_folders(final_path, final_path_full)


sample = 0          

while(True):
    ret, frame = cam.read()

    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    if detector == "ssd":  # SSD
        face_roi, processed_frame = detect_face_ssd(network, frame)
    else:  
        face_roi, processed_frame = detect_face(face_detector, frame)

    if face_roi is not None:
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sample = sample+1
            photo_sample = sample+starting_sample_number-1 if starting_sample_number>0 else sample
            image_name = person_name + "." + str(photo_sample) + ".jpg"
            cv2.imwrite(final_path + "/" + image_name, face_roi)  
            cv2.imwrite(final_path_full + "/" + image_name, frame)  
            print("=> photo " + str(sample))

            cv2.imshow("face", face_roi)


    cv2.imshow("Lendo a face do usuario", processed_frame)
    cv2.waitKey(1)

    if (sample >= max_samples):
        break

print ("Completo!")
cam.release()
cv2.destroyAllWindows()