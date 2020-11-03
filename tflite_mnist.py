import sys
sys.path.insert(0, "/usr/lib64/python3.5m/site-packages")
import numpy as np
import cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import time

# with open("labelmap.txt", 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

interpreter = tflite.Interpreter(model_path="mnist.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(output_details[0])
# print(input_details, output_details)
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
print('w x h', width, height)

cap_send = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

class Thread(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)
    str_signal = pyqtSignal(str)

    def run(self):
        if cap_send.isOpened():
            # ret, frame = cap_send.read()
            # cv2.imwrite('output.png', frame)
            while True:
                ret, frame = cap_send.read()
                
                # frame = cv2.flip(frame, 1)
                
                data = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                data = cv2.resize(data, ( 28, 28), fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
                
                data = np.array(data, dtype=np.float32)
                data = (data / 3) / 255
                data = np.expand_dims(data,0).astype(np.float32)
                # print("Input data shape:", data.shape)
                # print("Input data type:", data.dtype)
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                # print(np.argmax(output_data))
                end_time = time.time()
                self.str_signal.emit('result: '+str(np.argmax(output_data))+', time: '+str(end_time-start_time))

                rgbImage = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
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
        self.result_label = QtWidgets.QLabel()
        self.label = QtWidgets.QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)

        self.result_label.resize(300, 100)
        self.result_label.setFont(QtGui.QFont('Times', 15))
        self.result_label.setStyleSheet("QLabel { color : red; }") 
        # self.show_image()
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        th = Thread(self)
        th.str_signal.connect(self.setText)
        th.changePixmap.connect(self.setImage)
        th.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def setText(self, text):
        # split_text = text.split(",")
        # if (len(split_text) > 1):
        #     self.result_label.setText(split_text[0])
        #     self.response_label.setText(split_text[1])
        # else:
            self.result_label.setText(text)

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