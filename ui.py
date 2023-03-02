# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: ui.py.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
# -*- coding: utf-8 -*-
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from model.unet_model import UNet
import numpy as np

# 窗口主类
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('基于Unet的皮肤疾病病灶区域分割')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        # # 初始化视频读取线程
        self.origin_shape = ()
        # 加载网络，图片单通道，分类为1。
        net = UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model_skin.pth', map_location=device))  # todo 模型位置
        # 测试模式
        net.eval()
        self.model = net

        net2 = UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        net2.to(device=device)
        # 加载模型参数
        net2.load_state_dict(torch.load('best_model.pth', map_location=device))  # todo 模型位置
        # 测试模式
        net2.eval()
        self.model2 = net2

        self.initUI()

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("基于UNET的医学影像分割展示程序——皮肤病")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        #   2
        # 图片检测子界面
        #
        # # # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget2 = QWidget()
        img_detection_layout2 = QVBoxLayout()
        img_detection_title2 = QLabel("基于UNET的医学影像分割展示程序——眼球")
        img_detection_title2.setFont(font_title)
        mid_img_widget2 = QWidget()
        mid_img_layout2 = QHBoxLayout()
        self.left_img2 = QLabel()
        self.right_img2 = QLabel()
        self.left_img2.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img2.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img2.setAlignment(Qt.AlignCenter)
        self.right_img2.setAlignment(Qt.AlignCenter)
        mid_img_layout2.addWidget(self.left_img2)
        mid_img_layout2.addStretch(0)
        mid_img_layout2.addWidget(self.right_img2)
        mid_img_widget2.setLayout(mid_img_layout2)
        up_img_button2 = QPushButton("上传图片")
        det_img_button2 = QPushButton("开始检测")
        up_img_button2.clicked.connect(self.upload_img2)
        det_img_button2.clicked.connect(self.detect_img2)
        up_img_button2.setFont(font_main)
        det_img_button2.setFont(font_main)
        up_img_button2.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button2.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout2.addWidget(img_detection_title2, alignment=Qt.AlignCenter)
        img_detection_layout2.addWidget(mid_img_widget2, alignment=Qt.AlignCenter)
        img_detection_layout2.addWidget(up_img_button2)
        img_detection_layout2.addWidget(det_img_button2)
        img_detection_widget2.setLayout(img_detection_layout2)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用医学影像语义分割系统\n\n 作者：罗森')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/about-me.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        about_img.setScaledContents(True)

        # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        # label_super = QLabel()  # todo 更换作者信息
        # label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>或者你可以在这里找到我-->肆十二</a>")
        # label_super.setFont(QFont('楷体', 16))
        # label_super.setOpenExternalLinks(True)
        # label_super.setOpenExternalLinks(True)
        # label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        # about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '皮肤病图片分割')
        self.addTab(img_detection_widget2, '眼球图片分割')
        self.addTab(about_widget, '关于作者')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        self.setTabIcon(2, QIcon('images/UI/lufei.png'))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    def upload_img2(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img2.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img2.setPixmap(QPixmap("images/UI/right.jpeg"))


    '''
    ***检测图片***
    '''

    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        img = cv2.imread(source)
        origin_shape = img.shape
        # print(origin_shape)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = self.model(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        im0 = cv2.resize(pred, self.origin_shape)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    def detect_img2(self):
        model2 = self.model2
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        img = cv2.imread(source)
        origin_shape = img.shape
        # print(origin_shape)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = model2(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        im0 = cv2.resize(pred, self.origin_shape)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
        self.right_img2.setPixmap(QPixmap("images/tmp/single_result.jpg"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
