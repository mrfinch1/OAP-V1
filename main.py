import cv2
import numpy as np
cap = cv2.VideoCapture("videonuz.mp4")
#cap = cv2.VideoCapture(0) Web kameranızdan görüntü almak için bu kodu kullanın
car_cascade = cv2.CascadeClassifier("cars.xml")
human_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
while 1:
    ret,frame = cap.read()
    resize = frame[300:800,300:1500]
    blur = cv2.GaussianBlur(resize,(5,5),0)
    hsv = cv2.cvtColor(resize,cv2.COLOR_BGR2HSV)
    #-----------------------------------
    #lower = np.array([0,0,130])
    #upper = np.array([0,95,225])  
    #-----------------------------------(beyaz şerit)
    lower = np.array([0,102,51])
    upper = np.array([255,255,255])
    mask = cv2.inRange(hsv,lower,upper)
    #-----------------------------------(sarı şerit)
    edges = cv2.Canny(mask,75,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,30,maxLineGap = 50)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray,1.1,1)
    human = human_cascade.detectMultiScale(gray,1.1,1)
    for(x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    for(x,y,w,h) in human:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    if lines is not None:
        for line in lines:
            x,y,w,h = line[0]
            cv2.line(blur,(x,y),(w,h),(0,255,0),4)
    if not ret:
        cap = cv2.VideoCapture("video.mp4")
        continue
    cv2.imshow("mask",mask)
    cv2.imshow("blur",blur)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(25)
    if key == 27:
        break
cap.relase()
cv2.DestroyAllWindows()