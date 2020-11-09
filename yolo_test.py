import sys
sys.path.insert(0, "/usr/lib64/python3.5m/site-packages")
import numpy as np
import cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import threading
import time

with open("coco_classes.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
# if labels[0] == '???':
#     del(labels[0])

interpreter = tflite.Interpreter(model_path="yolov4-tiny-416.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details, output_details)
print(output_details)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('w x h', width, height)
input_mean = 255
input_std = 255

boxes = []
classes = []
scores = []
class_names = []
image = None

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

anchors = read_anchors("yolo_anchors.txt")

with open("coco_classes.txt", 'r') as f:
    class_boxes = np.array([line.strip() for line in f.readlines()])

# cap_send = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
image = cv2.imread('parallelparking.jpg')

imH, imW, _ = image.shape
print(imW, imH)
scaleW = imW / width
scaleH = imH / height


# ret, frame = cap_send.read()
# image = frame.copy()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_data = cv2.resize(image_rgb, (width, height))
image_data = image_data / 255.
# input_data = np.expand_dims(image_resized, axis=0)
# input_data = (np.float32(input_data) - input_mean) / input_std #float32

images_data = []
for i in range(1):
    images_data.append(image_data)
images_data = np.asarray(images_data).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'],images_data)
interpreter.invoke()

pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = [416,416]):
  boxes = []
  pred_conf = []
  pred = []

  for id, value in enumerate(scores[0]):
    # print(np.max(value))
    if np.max(value) > score_threshold:
        # bbbb = box_xywh[0][id].reshape((scores.shape[0], -1, class_boxes.shape[-1]))
        # print(bbbb)
        # maxPred = np.argmax(pred_conf[0][0])
        pred_conf = scores[0][id].reshape((scores.shape[0], -1, class_boxes.shape[-1]))
        maxPred = pred_conf[0][0].max()
        # j = np.unravel_index(pred_conf[0][0].argmax(), pred_conf[0][0].shape)
        # print(maxPred)
        # indices = np.where(pred_conf[0][0] == maxPred)
        # indices = np.where(pred_conf[0][0] >= 0.002)
        if maxPred > 0.4:
          box_xy = np.array([box_xywh[0][id][0], box_xywh[0][id][1]])
          box_wh = np.array([box_xywh[0][id][2], box_xywh[0][id][3]])
          box_yx = box_xy[..., ::-1]
          box_hw = box_wh[..., ::-1]
          box_mins = (box_yx - (box_hw / 2.)) / np.array([416, 416])
          box_maxes = (box_yx + (box_hw / 2.)) / np.array([416, 416])
          box = np.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
          ], axis=-1)
          print(box)
          boxes.append(box)
          indices = np.array(pred_conf[0])
          offset = 0
          # print('default', scores[0][id])
          # print(maxPred, indices)
          counter = -1
          # for val in pred_conf[0][0]:
          #   counter += 1
          #   if val > 0.001:
          #     print(val, counter)
          for val in indices[0]:
            # print(index)
            # print(pred_conf[0][0][index], maxPred)
            counter += 1
            if val == maxPred:
            # # if pred_conf[0][0][index] == maxPred:
            #   ind = np.where(val == maxPred)
              offset = counter
              # print(val, maxPred, counter)
          print(class_boxes[offset])
          pred.append([
            offset,
            maxPred,
          ])
    #   class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    #   pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

  return (boxes, pred)
  # scores_max = tf.math.reduce_max(scores, axis=-1)

boxes, pred = filter_boxes(pred[0], pred[1], score_threshold=0.1)
# print('boxes', boxes)
# boxes = pred[0]
editImage = image.copy()
# imH, imW = (1, 1)
imH, imW, _ = image.shape

npBox = np.array(boxes)

for i in range(len(boxes)):
    print(boxes[i], class_boxes[pred[i][0]])

  # if len(boxes) > i:
    ymin = int(max(1,(boxes[i][0] * imH)))
    xmin = int(max(1,(boxes[i][1] * imW)))
    ymax = int(min(imH,(boxes[i][2] * imH)))
    xmax = int(min(imW,(boxes[i][3] * imW)))

    object_name = class_boxes[pred[i][0]] # Look up object name from "labels" array using class index
    label = '%s: %d%%' % (object_name, int(pred[i][1]*100)) # Example: 'person: 72%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) # Get font size
    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
    cv2.rectangle(editImage, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
    cv2.putText(editImage, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Draw label text
    
    cv2.rectangle(editImage, (xmin,ymin), (xmax,ymax), (10, 255, 0), 1)
  
cv2.imwrite("output.jpg", editImage)