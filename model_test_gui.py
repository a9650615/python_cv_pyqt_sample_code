import time
import sys
sys.path.insert(0, "/usr/lib64/python3.5m/site-packages")
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import cv2
from subprocess import Popen, check_output, check_call, PIPE, call
import numpy as np

your_exe_file_address = "./Mnist" # example
your_command = 'mnist_model.tflite'
your_module_address = "five.bin" # example
first_run = True

cap_send = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

process = Popen([your_exe_file_address], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True) #, shell=True

def saveImgToBin(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, ( 28, 28), fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
    
    data = np.array(image, dtype=np.float32)
    data = (data / 3) / 255
    data.tofile("input.bin")

class Thread(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)
    str_signal = pyqtSignal(str)
    getRes = False

    def run(self):
        if cap_send.isOpened():
            while True:
                ret, frame = cap_send.read()

                height, width, channels = frame.shape
                cropSize = height
                offsetW = int((width - cropSize) / 2)
                y = 0
                crop_img = frame[offsetW:cropSize, y:height]
                # cv2.imwrite('save.png', crop_img)

                saveImgToBin(crop_img)
                global first_run
                if first_run == True:
                    print("send")
                    process.stdin.write(b'./input.bin\r\n')
                    process.stdin.flush()
                    first_run = False
                
                # tf
                output = process.stdout.readline()

                if output == '' and process.poll() is not None:
                    break
                if output:
                    if output.strip().find(b"Result") != -1:
                        self.getRes = True

                    self.str_signal.emit(output.strip().decode("utf-8") )
                if self.getRes == True:
                    process.stdin.write(b'./input.bin\r\n')
                    process.stdin.flush()
    
                rc = process.poll()
                #end tf
                rgbImage = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                self.image = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = self.image.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class DisplayImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DisplayImageWidget, self).__init__(parent)

        self.image_frame = QtWidgets.QLabel()
        self.result_label = QtWidgets.QLabel()
        self.response_label = QtWidgets.QLabel()
        self.result_label.resize(300, 100)
        self.result_label.setFont(QtGui.QFont('Times', 15))
        self.response_label.setFont(QtGui.QFont('Times', 15))
        self.result_label.setStyleSheet("QLabel { color : red; }") 

        self.label = QtWidgets.QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.response_label)
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

    # @QtCore.pyqtSlot(QtWidgets.QLabel)
    def setText(self, text):
        split_text = text.split(",")
        if (len(split_text) > 1):
            self.result_label.setText(split_text[0])
            self.response_label.setText(split_text[1])
        else:
            self.result_label.setText(text)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    display_image_widget = DisplayImageWidget()
    display_image_widget.show()
    sys.exit(app.exec())
    cap_send.release()