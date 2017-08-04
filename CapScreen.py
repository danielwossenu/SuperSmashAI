import numpy as np
import cv2
import tensorflow as tf
from PIL import ImageGrab
a = np.array([1,2])


while True:
    screen = np.array(ImageGrab.grab(bbox=(20,89,621,530)))
    damage = screen[-70:-20, 10:270, :]
    damage_p1 = screen[-70:-20, 10:115, :]
    damage_p2 = screen[-70:-20, 165:270, :]

    gray = cv2.cvtColor(damage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            x += 10
            y += 370

            if h > 30 and h < 40 and w > 10 and w < 30:
                print(w)
                cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
    cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break