import numpy as np
import cv2
import tensorflow as tf
from PIL import ImageGrab
a = np.array([1,2])


while True:
    screen = np.array(ImageGrab.grab(bbox=(20,89,621,530)))
    # printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
    cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break