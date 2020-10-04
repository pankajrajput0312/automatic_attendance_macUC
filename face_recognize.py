
def testing():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy
    import cv2
    from PIL import Image
    import PIL
    import os
    import shutil
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        './haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("trained_model.yml")

    while(True):
        ret, frame = cap.read()
        if(ret == False):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (51, 255, 51), 3)
            Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            confidence = 100-confidence
            name = str(Id)+" with "+str(confidence)
            print(name)
            if(confidence > 50):
                text = str(Id)+" marked "
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#               image = cv2.putText(image, 'OpenCV', org, font,
#                        fontScale, color, thickness, )
            if(confidence < 50):
                text = " not marked"
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("frame", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


testing()
