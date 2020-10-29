import sys
sys.path.insert(0, "/usr/lib64/python3.5m/site-packages")
import numpy as np
import cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

with open("labelmap.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

interpreter = tflite.Interpreter(model_path="custom_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details, output_details)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('w x h', width, height)
input_mean = 127.5
input_std = 127.5

cap_send = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

class Thread(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)

    def run(self):
        if cap_send.isOpened():
            # ret, frame = cap_send.read()
            # cv2.imwrite('output.png', frame)
            while True:
                ret, frame = cap_send.read()
                frame = cv2.flip(frame, 1)
                
                image = frame
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                imH, imW, _ = image.shape
                image_resized = cv2.resize(image_rgb, (width, height))
                input_data = np.expand_dims(image_resized, axis=0)
                input_data = (np.float32(input_data) - input_mean) / input_std #float32
                
                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()

                boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
                classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
                scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

                for i in range(len(scores)):
                  if ((scores[i] > 0.2) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

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