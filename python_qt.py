import sys
sys.path.insert(0, "/usr/lib64/python3.5m/site-packages")
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow

app = QApplication([])
win = QMainWindow()
win.setWindowTitle("window title")
win.setFixedWidth(300)
label = QLabel('Hello world!!', win)
# label.show()
win.show()
sys.exit (app.exec())