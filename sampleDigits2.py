import numpy as np
from collections import Counter
import cv2
import tensorflow as tf
from PIL import ImageGrab
import sys
import os.path

orig_screen = cv2.imread('previous.png')
screen = orig_screen.copy()
damage = screen[-70:-20, 10:270, :]
damage_p1 = screen[-70:-20, 10:115, :]
damage_p2 = screen[-70:-20, 165:270, :]

gray = cv2.cvtColor(damage, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (1, 1), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('window', blur)
cv2.waitKey(0)

for cnt in contours:
    screen = orig_screen.copy()
    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)
        orig_x = x
        orig_y = y
        x += 10
        y += 370

        if h > 30 and h < 40 and w > 10 and w < 30:
            print(w)
            cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[orig_y:orig_y + h, orig_x:orig_x + w]
            roismall = cv2.resize(roi, (10, 10), interpolation = cv2.INTER_CUBIC)
            cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(0)
            print(key)