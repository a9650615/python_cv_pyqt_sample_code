import sys
sys.path.insert(0, "/usr/lib64/python3.5m/site-packages")
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import cv2
import sys
import numpy as np

#cap_send = cv2.VideoCapture('videotestsrc ! video/x-raw ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
cap_send = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

class Thread(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)

    def run(self):
        if cap_send.isOpened():
            # ret, frame = cap_send.read()
            # cv2.imwrite('output.png', frame)
            while True:
                ret, frame = cap_send.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (11, 11), 0)
                edged = cv2.Canny(blurred, 30, 150)
                (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                coins = frame.copy()
                cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)

                rgbImage = cv2.cvtColor(coins, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                self.image = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = self.image.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class DisplayImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DisplayImageWidget, self).__init__(parent)

        self.button = QtWidgets.QPushButton('Show picture')
        # self.button.clicked.connect(self.show_image)
        self.image_frame = QtWidgets.QLabel()

        self.label = QtWidgets.QLabel(self)
        self.label.move(280, 120)
        self.label.resize(640, 480)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))


    @QtCore.pyqtSlot()
    def show_image(self):
        self.image = cv2.imread('image.jpg')
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    display_image_widget = DisplayImageWidget()
    display_image_widget.show()
    sys.exit(app.exec())
    cap_send.release()