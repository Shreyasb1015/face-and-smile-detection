import numpy as np
import cv2 as cv


face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_smile.xml')


cap=cv.VideoCapture(0,cv.CAP_DSHOW)

def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5) # type: ignore
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        smiles=smile_cascade.detectMultiScale(roi_gray,1.8,20)
        for(sx,sy,sw,sh) in smiles:
            cv.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,0,0),5) # type: ignore
    return frame

while True:
    ret,frame=cap.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    result=detect(gray,frame)
    
    cv.imshow('smile detection',result)
    
    if cv.waitKey(1)==ord('a'):
       break
cap.release()
cv.destroyAllWindows()
    