import numpy as np
import cv2
from time import sleep
import imutils

cap = cv2.VideoCapture(0)

windowSize = 450
frame_counter = 0

cv2.namedWindow('progressive bg')
cv2.namedWindow('frame')
cv2.moveWindow('frame', 0,0)
cv2.moveWindow('progressive bg', windowSize,0)

fgbg = cv2.createBackgroundSubtractorMOG2()
cv2.namedWindow('cv2.createBackgroundSubtractorMOG2')
cv2.moveWindow('cv2.createBackgroundSubtractorMOG2', windowSize * 2, 0)
while cap.isOpened() :
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    frame = imutils.resize(frame, width=windowSize)
    blank_image = np.zeros((frame.shape), np.uint8)
    fgmog2 = fgbg.apply(frame)
    kernelmog2 = np.ones((3, 3), np.uint8)
    fgmaskmog2 = cv2.erode(fgmog2, kernelmog2, iterations=1)
    resmog2 = cv2.bitwise_and(frame, frame, mask=fgmaskmog2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.flip(frame, 1)  # flip the frame horizontally
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if frame_counter == 0:
        print("Here we go!")
        avg = gray.copy().astype("float")
        continue
    cv2.accumulateWeighted(gray, avg, 0.03)
    diff = cv2.absdiff(avg.astype("uint8"), gray)
    thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    res = cv2.bitwise_and(frame, frame, mask=fgmask.copy())
    (_,contours,_) = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        blank_image[y:y+h,x:x+w] = frame[y:y+h,x:x+w]
    cv2.imshow('rect extractor',blank_image)
    cv2.imshow('progressive bg',res)
    cv2.imshow('frame',frame)
    cv2.imshow('cv2.createBackgroundSubtractorMOG2',resmog2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
