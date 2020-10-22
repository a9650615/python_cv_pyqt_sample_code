from os import walk
import sys
import numpy as np
import cv2
from PIL import Image

def convertImageForMobileNetQuant(filename, outfilename):
    pil_image = Image.open(filename)
    pil_image = pil_image.resize((28,28), Image.ANTIALIAS)
    image = cv2.imread(filename) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, ( 28, 28), fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
    pil_data = np.array(pil_image, dtype=np.float32)
    data = np.array(image, dtype=np.float32)
    print(pil_data)
    print("-----")
    print( data)
    data = (data / 3) / 255
    data.tofile(outfilename)

if __name__ == '__main__':
    convertImageForMobileNetQuant(sys.argv[1], sys.argv[2])

    