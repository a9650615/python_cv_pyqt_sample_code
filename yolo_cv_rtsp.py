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
    class_boxes = np.array([line.strip() for line in f.readlines()])

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
image = None

cap_send = cv2.VideoCapture('rtspsrc do-rtcp=TRUE location=rtspt://admin:123456@192.168.0.104:554/stream2  ! rtph264depay ! h264parse ! decodebin ! autovideoconvert ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
# cap_send = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = [416,416]):
  boxes = []
  pred_conf = []
  pred = []
  position = []

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
        if maxPred > 0.25:
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
        #   print(box)
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
            counter += 1
            if val == maxPred:
            # # if pred_conf[0][0][index] == maxPred:
            #   ind = np.where(val == maxPred)
              offset = counter
              break
            
          class_name = class_boxes[offset]
          pred.append([
            offset,
            maxPred,
            class_name,
          ])
    #   class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
    #   pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])

  return (boxes, pred)

def tf():
    global boxes, classes, scores, image, width, height

    counter = 1

    while True:
        counter = counter + 1
        counter = counter % 10

        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_data = cv2.resize(image_rgb, (width, height))
            image_data = image_data / 255.
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'],image_data)
            interpreter.invoke()
            
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            
            boxes, classes = filter_boxes(pred[0], pred[1], score_threshold=0.2)

class Thread(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)

    def run(self):
        global boxes, classes, scores, image

        if cap_send.isOpened():
            # ret, frame = cap_send.read()
            # cv2.imwrite('output.png', frame)
            while True:
                boxes_draw = np.copy(boxes)
                classes_draw = np.copy(classes)
                ret, frame = cap_send.read()
                image = frame.copy()
                imH, imW, _ = image.shape # todo: fix

                if len(classes_draw) > 0:
                    for i in range(len(classes_draw)):
                        # if ((scores[i] > 0.1) and (scores[i] <= 1.0)):
                            # Get bounding box coordinates and draw box
                            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        if len(boxes_draw) > i:
                            ymin = int(max(1,(boxes_draw[i][0] * imH)))
                            xmin = int(max(1,(boxes_draw[i][1] * imW)))
                            ymax = int(min(imH,(boxes_draw[i][2] * imH)))
                            xmax = int(min(imW,(boxes_draw[i][3] * imW)))
                            
                            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 1)

                            # Draw label
                            object_name = classes_draw[i][2] # Look up object name from "labels" array using class index
                            label = '%s: %d%%' % (object_name, int(float(classes_draw[i][1])*100)) # Example: 'person: 72%'
                            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2) # Get font size
                            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Draw label text

                rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # rgbImage = image.copy()
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                self.image = rgbImage
                self.image = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = self.image.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

class DisplayImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DisplayImageWidget, self).__init__(parent)
        self.image_frame = QtWidgets.QLabel()
        self.label = QtWidgets.QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)

        # self.show_image()
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        t = threading.Thread(target = tf)
        t.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    # @QtCore.pyqtSlot()
    def show_image(self):
      img = cv2.imread('sachin.jpg')
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
          img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]
          eyes = eye_cascade.detectMultiScale(roi_gray)
          for (ex,ey,ew,eh) in eyes:
              cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

      self.image = img
      self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
      self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    display_image_widget = DisplayImageWidget()
    display_image_widget.show()
    sys.exit(app.exec())