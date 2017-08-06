import numpy as np
from collections import Counter
import cv2
import tensorflow as tf
from PIL import ImageGrab
import sys
from OCR_Damage import digitOCR
import os.path
a = np.array([1,2])

# https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python

if os.path.isfile('damagedigits.data'):
    samples = np.loadtxt('damagedigits.data')
    responses = np.loadtxt('damagelabels.data').tolist()
else:
    samples = np.empty((0, 100))
    responses = []

OCR = digitOCR()
keys = [i for i in range(48, 58)]
previous_image = None

while True:
    orig_screen = np.array(ImageGrab.grab(bbox=(20,89,621,530)))
    screen = orig_screen.copy()
    damage = screen[-70:-20, 10:270, :]
    damage_p1 = screen[-70:-20, 10:115, :]
    damage_p2 = screen[-70:-20, 165:270, :]

    gray = cv2.cvtColor(damage, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (1, 1), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    image, contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    exit_image = False
    damage1position = []
    damage1value = []
    damage2position = []
    damage2value = []

    for cnt in contours:
        # screen = orig_screen.copy()

        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)
            orig_x = x
            orig_y = y
            x += 10
            y += 370



            if h > 30 and h < 40 and w > 10 and w < 30:
                # print(w)
                cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi = thresh[orig_y:orig_y + h, orig_x:orig_x + w]
                roismall = cv2.resize(roi, (10, 10), interpolation = cv2.INTER_CUBIC)
                prediction = OCR.predict(roismall.reshape((1, 100)))
                # print(prediction)
                if orig_x < 115 and y>381 and orig_x > 25:
                    damage1position.append([orig_x,prediction])
                    # damage1value.append(OCR.predict(roismall.reshape((1, 100))))
                elif orig_x > 165 and y>381:
                    damage2position.append([orig_x, prediction])
                    # damage1value.append(OCR.predict(roismall.reshape((1, 100))))

    damage1position = sorted(damage1position, reverse=True)
    damage2position = sorted(damage2position, reverse=True)
    d1=0
    d2=0
    for i,d in enumerate(damage1position):
        d1 += d[1] * 10**i
    for i, d in enumerate(damage2position):
        d2 += d[1] * 10 ** i
    print(d1,d2)
    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))

    key = cv2.waitKey(0)
    # print(key)

    if key == 27:  # (escape to quit)
        # sys.exit()
        exit_image = True
        break
    # print(Counter(responses))
    previous_image = orig_screen
    if exit_image:
        break

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

# np.savetxt('damagedigits.data', samples)
# np.savetxt('damagelabels.data', responses)


    # printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))

    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break