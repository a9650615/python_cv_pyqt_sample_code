from os import walk
import sys
import numpy as np
import cv2
from PIL import Image

def convertImageForMobileNetQuant(filename, outfilename):
    original_image = cv2.imread(filename)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    images_data.tofile(outfilename)

if __name__ == '__main__':
    convertImageForMobileNetQuant(sys.argv[1], sys.argv[2])

    