import sys
import cv2
import numpy as np
import time
from typing import Union
from Stabilization import Stabilizer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QFileDialog, QInputDialog, QLineEdit, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

class VideoThread(QThread):

    original_frame = pyqtSignal(QImage)

    def __init__(self, src_url: str, FPS: int):
        super(VideoThread, self).__init__()
        self.src_url = src_url
        self.FPS = FPS

    def img_2_qimage(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        bytes_per_line = c * w
        img_qt_format = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return img_qt_format
    
    def run(self):
        video_capture = cv2.VideoCapture(self.src_url)

        while True:
            ret, frame = video_capture.read()
            if ret:
                start = time.time()
                img_qt_format = self.img_2_qimage(frame)
                self.original_frame.emit(img_qt_format)
                end = time.time()
                
                sleep_time = 1/self.FPS - (end - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                break
        video_capture.release()

class StabilizeThread(QThread):

    stabilized_frame = pyqtSignal(QImage)

    def __init__(self, src_url: str, FPS: int):
        super(StabilizeThread, self).__init__()
        self.src_url = src_url
        self.stabilizer = Stabilizer()
        self.FPS = FPS

    def img_2_qimage(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        bytes_per_line = c * w
        img_qt_format = QtGui.QImage(img.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return img_qt_format

    def run(self):
        video_capture = cv2.VideoCapture(self.src_url)

        # first time init
        ret, frame = video_capture.read()
        if ret:
            self.stabilizer.init(frame)
            self.stabilized_frame.emit(self.img_2_qimage(frame))

        while True:
            ret, frame = video_capture.read()
            if ret:
                start = time.time()
                stabilized_img = self.stabilizer.stabilize(frame)
                stabilized_qtimage = self.img_2_qimage(stabilized_img)
                self.stabilized_frame.emit(stabilized_qtimage)
                end = time.time()

                sleep_time = 1/self.FPS - (end - start)
                if sleep_time > 0:
                    time.sleep(sleep_time)                
            else:
                break
        video_capture.release()

class Application(QMainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.__init_window()

    def __init_window(self):
        self.setWindowTitle('Video Stabilization')
        self.left = 100
        self.top = 100
        self.width = 1024
        self.height = 500
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.FPS = 24

        self.original_frame = QLabel(self)
        self.original_frame.resize(self.width/2, 480)

        self.processed_frame = QLabel(self)
        self.processed_frame.move(self.width/2, 0)
        self.processed_frame.resize(self.width/2, 480)

        self.video_stream = None
        self.stabilize_stream = None

        self.__init_menu()

    def __init_menu(self):
        open_from_file = QAction('&Open from file', self)
        open_from_file.setShortcut('Ctrl+O')
        open_from_file.setStatusTip('Open a video from file')
        open_from_file.triggered.connect(self.open_file_dialog_callback)

        open_from_url = QAction('&Open from URL', self)
        open_from_url.setShortcut('Ctrl+Shift+O')
        open_from_url.setStatusTip('Open a video from streaming URL')
        open_from_url.triggered.connect(self.open_url_dialog_callback)

        open_from_camera = QAction('&Open from Camera', self)
        open_from_camera.setShortcut('Ctrl+Alt+O')
        open_from_camera.setStatusTip('Open a video from streaming URL')
        open_from_camera.triggered.connect(self.open_camera_dialog_callback)

        quit_app = QAction('&Quit', self)
        quit_app.setShortcut('Alt+F4')
        quit_app.setStatusTip('Quit')
        quit_app.triggered.connect(self.close_application)

        show_about = QAction('&About', self)
        show_about.setStatusTip('About us')
        show_about.triggered.connect(self.__show_about)

        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(open_from_file)
        fileMenu.addAction(open_from_url)
        fileMenu.addAction(open_from_camera)
        fileMenu.addSeparator()
        fileMenu.addAction(quit_app)

        mainMenu.addAction(show_about)

        play_action = QAction(QIcon('play.jpg'), '&Play', self)
        play_action.setShortcut('Space')
        play_action.setStatusTip('Play')
        play_action.triggered.connect(self.play_video)

        toolbar = self.addToolBar('Exit')
        toolbar.addAction(play_action)

    def __show_about(self):

        about = 'Student ID: 1612348\n- Name: Ly Vinh Loi\n- Email: vinhloiit1327@gmail.com\nStudent ID: 1612357\n- Name: Tran Tan Luan\n- Email: luantranhcmus@gmail.com'
        message = QMessageBox()
        message.setText(about)
        message.exec()

    def open_file_dialog_callback(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, ok = QFileDialog.getOpenFileName(self, "Select file", "","AVI File (*.avi);;MP4 Files (*.mp4);;All Files (*)", options=options)
        if ok:
            print(f'Open video from file: {file_name}')
            self.setup_stream(file_name)

    def open_camera_dialog_callback(self):
        cam_index, ok = QInputDialog.getInt(self, "Select Camera Index", "Enter Camera Index (default = 0)", QLineEdit.Normal)
        if ok:
            print(f'Open camera from Camera index = {cam_index}')
            self.setup_stream(cam_index)

    def open_url_dialog_callback(self):
        url, ok = QInputDialog.getText(self, "Stream URL", "Enter streaming URL", QLineEdit.Normal)
        if ok:
            print(f'Open video from {url}')
            self.setup_stream(url)

    def setup_stream(self, source):
        self.video_stream = VideoThread(source, self.FPS)
        self.video_stream.original_frame.connect(self.update_original_frame)

        # self.stabilize_stream = VideoThread('0_stable.avi', self.FPS)
        # self.stabilize_stream.original_frame.connect(self.update_processed_frame)

        self.stabilize_stream = StabilizeThread(source, self.FPS)
        self.stabilize_stream.stabilized_frame.connect(self.update_processed_frame)

    def play_video(self):
        if self.video_stream and self.stabilize_stream:
            if self.video_stream.isRunning():
                self.video_stream.quit()
            if self.stabilize_stream.isRunning():
                self.stabilize_stream.quit()
            
            self.video_stream.wait()
            self.stabilize_stream.wait()

            self.video_stream.start()
            self.stabilize_stream.start()
        else:
            self.__show_error_select_video_first()

    def close_application(self):
        sys.exit()

    def __show_error_select_video_first(self):
        text = 'You should select video source first from menu File'
        message = QMessageBox()
        message.setText(text)
        message.exec()

    @pyqtSlot(QImage)
    def update_original_frame(self, frame):
        frame = frame.scaled(self.width / 2, 480, Qt.KeepAspectRatio)
        self.original_frame.setPixmap(QPixmap.fromImage(frame))

    @pyqtSlot(QImage)
    def update_processed_frame(self, frame):
        frame = frame.scaled(self.width / 2, 480, Qt.KeepAspectRatio)
        self.processed_frame.setPixmap(QPixmap.fromImage(frame))


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Application()
    window.show()
    app.exec_()