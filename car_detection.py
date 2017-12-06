import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("car_detection.xml")

img = cv2.imread("car_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imwrite('detected_cat.jpg', img)
print("Success")
cv2.waitKey()
