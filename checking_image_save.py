import matplotlib.pyplot as plt
import pandas as pd
import numpy
import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
count = 0
name = input("enter name")
Id = input(" enter id")
while(True):
    ret, frame = cap.read()
    if(ret == False):
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (51, 255, 51), 3)
        count += 1
        img_name_path = str(id)+"."+str(count)+".png"
        offset = 10
        if(count % 50 == 0):
            print(img_name_path)

        status = cv2.imwrite(img_name_path, gray[x:x+w, y:y+h])
        print(status)
        cv2.imshow("frame", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif(count > 200):
        break

cap.release()
cv2.destroyAllWindows()
