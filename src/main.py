import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

class VideoThread(QThread):

    change_pix_map = pyqtSignal(QImage)

    def __init__(self, src_url: str):
        super(VideoThread, self).__init__()
        self.src_url = src_url
    
    def run(self):
        video_capture = cv2.VideoCapture(self.src_url)
        while True:
            ret, frame = video_capture.read()
            if ret:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = rgb_img.shape
                bytes_per_line = c * w
                img_qt_format = QtGui.QImage(rgb_img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                p = img_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pix_map.emit(p)


class Application(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.__init_window()

        self.camera = VideoThread(0)
        self.camera.change_pix_map.connect(self.updateFrame)
        self.camera.start()

    def __init_window(self):
        self.setWindowTitle('Video Stabilization')
        self.left = 100
        self.top = 100
        self.width = 800
        self.height = 600
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label = QLabel(self)
        self.label.move(50, 50)
        self.label.resize(640, 480)

    @pyqtSlot(QImage)
    def updateFrame(self, frame):
        self.label.setPixmap(QPixmap.fromImage(frame))


if __name__ == '__main__':

    app = QApplication(sys.argv)

    window = Application()
    window.show()
    app.exec_()