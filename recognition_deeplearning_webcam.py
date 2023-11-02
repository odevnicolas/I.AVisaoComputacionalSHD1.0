import cv2
import numpy as np
import os
import pickle
import re
import face_recognition


from helper_functions import resize_video

pickle_name = "face_encodings_custom.pickle" 
max_width = 800          


data_encoding = pickle.loads(open(pickle_name, "rb").read())
list_encodings = data_encoding["encodings"]
list_names = data_encoding["names"]

def recognize_faces(image, list_encodings, list_names, resizing=0.25, tolerance=0.5):
  image = cv2.resize(image, (0, 0), fx=resizing, fy=resizing)

  img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  face_locations = face_recognition.face_locations(img_rgb)
  face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

  face_names = []     
  conf_values = []    
  for encoding in face_encodings:
  
    matches = face_recognition.compare_faces(list_encodings, encoding, tolerance=tolerance)
    name = "Usuário não cadastrado"
    
    face_distances = face_recognition.face_distance(list_encodings, encoding)
    best_match_index = np.argmin(face_distances) 
    if matches[best_match_index]:
      name = list_names[best_match_index] 
    face_names.append(name) 
    conf_values.append(face_distances[best_match_index])

  face_locations = np.array(face_locations)
  face_locations = face_locations / resizing
  return face_locations.astype(int), face_names, conf_values

def show_recognition(frame, face_locations, face_names, conf_values):

for face_loc, name, conf in zip(face_locations, face_names, conf_values):
    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

    if name == "Não identificado":
        rect_color = (0, 0, 255)  
    else:
        rect_color = (0, 255, 0)  

    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (20, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 4)

    if name != "Não identificado":
        conf = "{:.8f}".format(conf)
        cv2.putText(frame, conf, (x1, y2 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 255, 0), 1, lineType=cv2.LINE_AA)


  return frame

 
cam = cv2.VideoCapture(0)


while(True):
    ret, frame = cam.read()

    if max_width is not None:
        video_width, video_height = resize_video(frame.shape[1], frame.shape[0], max_width)
        frame = cv2.resize(frame, (video_width, video_height))

    face_locations, face_names, conf_values = recognize_faces(frame, list_encodings, list_names, 0.25)
    print(face_locations)
    processed_frame = show_recognition(frame, face_locations, face_names, conf_values)

    cv2.imshow("Lendo a face do usuário ", frame)
    cv2.waitKey(1)

print ("Completo!")
cam.release()
cv2.destroyAllWindows()