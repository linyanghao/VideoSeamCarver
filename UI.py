# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 02:07:44 2019

@author: JUNJIN
"""

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image
import numpy as np
from video import *
from targetArea import *
#主窗口类
class TabDemo(QTabWidget):
    """继承窗口类"""
    def __init__(self,parent=None):
        """初始化"""
        #控制窗口的大小
        super(TabDemo,self).__init__(parent)
        width = QDesktopWidget().availableGeometry().width() * 3 / 4
        height = width/2
        self.canvas_w = width*2/5
        self.canvas_h = width*2/5
        #tab1 tab2 tab3分别代表图片裁剪 物品消除 视频裁剪三个任务
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.addTab(self.tab1,"Tab 1")
        self.addTab(self.tab2,"Tab 2")
        self.addTab(self.tab3,"Tab 3")
        
        self.tab1UI()
        self.tab2UI()
        self.tab3UI()
        self.setGeometry(0, 0, self.canvas_w, height)
        self.setWindowTitle("Seam Carver")
        self.setMouseTracking(True)
    #tab1的UI交互
    def tab1UI(self):
        """图片裁剪窗口"""
        self.tab1.inforEdit = QLineEdit()
        #一些打开文件的按钮
        self.tab1.inforEdit.setText("Welcome!")
        self.tab1.textEdit = QLineEdit()
        self.tab1.FileButton = QPushButton("Open File")
        self.tab1.OKButton = QPushButton("OK")
        self.tab1.Cancle = QPushButton("Cancle")
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox.addWidget(self.tab1.textEdit)
        hbox.addWidget(self.tab1.FileButton)
        hbox.addWidget(self.tab1.OKButton)
        hbox.addWidget(self.tab1.Cancle)
        hbox.addStretch(1)
        hbox1 = QHBoxLayout()
        self.tab1.textEdit1 = QLineEdit()
        self.tab1.textEdit2 = QLineEdit()
        #设置长宽进行裁剪
        infor1 = QLabel('Width', self.tab1)
        infor2 = QLabel('Height',self.tab1)
        self.tab1.functionButton = QPushButton("Seam")
        hbox1.addWidget(infor1)
        hbox1.addWidget(self.tab1.textEdit1)
        hbox1.addWidget(infor2)
        hbox1.addWidget(self.tab1.textEdit2)
        hbox1.addWidget(self.tab1.functionButton)
        hbox1.addWidget(self.tab1.inforEdit)
        hbox1.addStretch(1)
        #Qlabel显示图片
        pixmap = QPixmap("1544672096862777.png")
        self.tab1.l1 = QLabel(self.tab1)
        self.tab1.l1.setPixmap(pixmap)
        #self.tab1.l1.setScaledContents(True)
        #self.tab1.l1.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)
        #self.tab1.l1.setFixedSize(self.canvas_w,self.canvas_h)
        vbox.addStretch(1)
        vbox.addWidget(self.tab1.l1)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox1)
        self.tab1.setLayout(vbox)
        #按钮触发函数
        self.setTabText(0,"image seam carver")
        self.tab1.FileButton.clicked.connect(self.showDialog)
        self.tab1.OKButton.clicked.connect(self.showImg)
        self.tab1.Cancle.clicked.connect(self.cleartext)
        self.tab1.functionButton.clicked.connect(self.seamer)
    
    def object_remove(self):
        """物品移除前后端交互"""
        """
        目标移除函数
        图片文件在filename = self.tab2.textEdit.text()
        选取的点数据在self.tab2.point = []
        将函数放在这里，跑出结果
        保存结果在本地
        然后在self.tab2.l1中以动图形式显示
        """
        #carver.Draw()
        points = len(self.tab2.point)
        if points != 2:
            self.tab2.inforEdit.setText("wrong number of points")
            self.clearPoint()
            return 
        point1,point2 = self.tab2.point[0],self.tab2.point[1]
        ux = point1[0]
        uy = point1[1]
        bx = point2[0]
        by = point2[1]
        self.tab2.inforEdit.setText("Please waiting")
        IMAGE_AS_VIDEO, SMALL_DATA = True, False
        print(uy)
        print(by)
        REMOVE_SEAM_TEST  = False
        AUGMENT_SEAM_TEST = False
        XY_SCALE_TEST     = False
        REMOVE_TARGET_TEST = True

        assert XY_SCALE_TEST + REMOVE_SEAM_TEST + AUGMENT_SEAM_TEST + REMOVE_TARGET_TEST == 1, "Wrong setting!"


    
        if IMAGE_AS_VIDEO:
            img   = Image.open(self.tab2.textEdit.text())
            img   = img.convert('RGB')
            img   = np.array(img)
            video = np.reshape(img, [1, *img.shape])
            print(video.shape)
        else:
            video = imageio.get_reader('golf.mov', 'ffmpeg')
            frames = []
            for i, frame in enumerate(video):
                frames.append(frame)
            video = np.array(frames)

        if SMALL_DATA:
            video = video[:10, ...]


    
        #carver.Draw()


        
        if REMOVE_TARGET_TEST:
            print("=========== REMOVE_TARGET_TEST ==============\n")
            carver = ContentRemover(video, indicator=AnyShapeIndicator([(0, i, j) for i in range(uy, by) for j in range(ux, bx)])) 
            frames = []

            removed_seams_count = 0
            while not carver.indicator.Empty():
                seam = carver.Solve()
                video_w_seam = carver.GenerateVideoWithSeam(seam)
                frames.append(video_w_seam[0])
                #imageio.mimsave(OUT_FOLDER+'/%s.gif' % removed_seams_count, video_w_seam)
                carver.RemoveSeam(seam)
                removed_seams_count += 1
                print('Currently %s seams removed' % (removed_seams_count))

            seams = carver.SolveK(removed_seams_count)
            for seam in seams:
                video_w_seam = carver.GenerateVideoWithSeam(seam)
                frames.append(video_w_seam[0])
                carver.AugmentSeam(seam)
            #结果文件保存的目录
            imageio.mimsave(OUT_FOLDER + "/remove_target_movie.gif", frames)        
            
            res = QMovie(OUT_FOLDER + "/remove_target_movie.gif")
            self.tab2.l1.setMovie(res)
            res.start()
        self.tab2.inforEdit.setText("Done!")    
        # stretch = VideoSeamCarver(res)
        # stretch.augment_hor(REMOVE_SEAMS_COUNT, save_hist=True, want_trans=False)
        
    
    def seamer(self):
        """图片裁剪前后端交互"""
        """
        seamer函数
        图片文件在filename = self.tab1.textEdit.text()
        将函数放在这里，跑出结果
        保存结果在本地
        然后在self.tab1.l1中显示以动图形式显示
        """
        self.tab1.inforEdit.setText("Please waiting")
        IMAGE_AS_VIDEO, SMALL_DATA = True, False

        REMOVE_SEAM_TEST  = False
        AUGMENT_SEAM_TEST = False
        XY_SCALE_TEST     = True

        assert XY_SCALE_TEST + REMOVE_SEAM_TEST + AUGMENT_SEAM_TEST == 1, "Wrong setting!"

        
        X_SEAMS_COUNT       = int(self.tab1.textEdit1.text()) - self.tab1.l1.w
        Y_SEAMS_COUNT       = int(self.tab1.textEdit2.text()) - self.tab1.l1.h
    
        if IMAGE_AS_VIDEO:
            img   = Image.open(self.tab1.textEdit.text())
            img   = img.convert('RGB')
            img   = np.array(img)
            video = np.reshape(img, [1, *img.shape])
            print(video.shape)
        else:
            video = imageio.get_reader('golf.mov', 'ffmpeg')
            frames = []
            for i, frame in enumerate(video):
                frames.append(frame)
                video = np.array(frames)

        if SMALL_DATA:
            video = video[:10, ...]
        carver = VideoSeamCarver(video)
        #carver.Draw()


        if XY_SCALE_TEST:
            print("=========== XY_SCALE_TEST ==============\n")
            print("=========== {} =========".format(
            "Shrink X" if X_SEAMS_COUNT < 0 else "Augment X"))
            x_scaled_video = carver.scale_hor(X_SEAMS_COUNT, save_hist=True)
            x_scaled_trans = np.transpose(x_scaled_video, (0,2,1,3))

            print("=========== {} =========".format(
            "Shrink Y" if Y_SEAMS_COUNT < 0 else "Augment Y"))
            y_scaler = VideoSeamCarver(x_scaled_trans)
            res = y_scaler.scale_hor(Y_SEAMS_COUNT, save_hist=True)
            res = np.transpose(res, (0,2,1,3))
            imageio.mimsave(OUT_FOLDER+ 'result.gif', res)
            res = QMovie(OUT_FOLDER+  'result.gif')
            self.tab1.l1.setMovie(res)
            res.start()
        self.tab1.inforEdit.setText("Done!")
    def seamer_video(self):
        """视频裁剪前后端交互"""
        """
        seamer函数对video的处理（应该是和seamer调用相同的外部函数）
        图片文件在filename = self.tab3.textEdit.text()
        将函数放在这里，跑出结果
        保存结果在本地
        然后在self.tab3.l2中显示以动图形式显示
        """
        self.tab3.inforEdit1.setText("Please waiting")
        XY_SCALE_TEST     = True

        assert XY_SCALE_TEST + REMOVE_SEAM_TEST + AUGMENT_SEAM_TEST == 1, "Wrong setting!"

        

    
        if IMAGE_AS_VIDEO:
            img   = Image.open(self.tab1.textEdit.text())
            img   = img.convert('RGB')
            img   = np.array(img)
            video = np.reshape(img, [1, *img.shape])
            print(video.shape)
        else:
            video = imageio.get_reader(self.tab3.textEdit3.text(), 'ffmpeg')
            frames = []
            for i, frame in enumerate(video):
                frames.append(frame)
                video = np.array(frames)
            _,h,w,_ = video.shape
            X_SEAMS_COUNT       = int(self.tab3.textEdit4.text()) - w
            Y_SEAMS_COUNT       = int(self.tab3.textEdit5.text()) - h
        if SMALL_DATA:
            video = video[:10, ...]
        carver = VideoSeamCarver(video)
        #carver.Draw()


        if XY_SCALE_TEST:
            print("=========== XY_SCALE_TEST ==============\n")
            print("=========== {} =========".format(
            "Shrink X" if X_SEAMS_COUNT < 0 else "Augment X"))
            x_scaled_video = carver.scale_hor(X_SEAMS_COUNT, save_hist=True)
            x_scaled_trans = np.transpose(x_scaled_video, (0,2,1,3))

            print("=========== {} =========".format(
            "Shrink Y" if Y_SEAMS_COUNT < 0 else "Augment Y"))
            y_scaler = VideoSeamCarver(x_scaled_trans)
            res = y_scaler.scale_hor(Y_SEAMS_COUNT, save_hist=True)
            res = np.transpose(res, (0,2,1,3))
            imageio.mimsave(OUT_FOLDER+ 'result.gif', res)
            res = QMovie(OUT_FOLDER+  'result.gif')
            self.tab1.l1.setMovie(res)
            res.start()
        self.tab3.inforEdit1.setText("Done!")
    def showDialog(self):
        """选择图片文件"""
        filename,  _ = QFileDialog.getOpenFileName(self, 'Open file', './')
        sender = self.sender()
        if filename[-3:] == "jpg" or filename[-3:] == "png" or filename[-4:] == "jpeg" or filename[-3:] == "bmp" :
            if sender == self.tab1.FileButton:
                self.tab1.textEdit.setText(filename)
            else:
                self.tab2.textEdit.setText(filename)
        else:
            pass
    
    def showDialog1(self):
        """选择视频文件 目前pyqt只支持显示gif"""
        filename,  _ = QFileDialog.getOpenFileName(self, 'Open file', './')
        
        if filename[-3:] == "gif" or filename[-3:] == 'mov':
            self.tab3.textEdit3.setText(filename)
        else:
            pass
        
    def showVideo(self):
        """显示gif"""
        filename = self.tab3.textEdit3.text()
        gif = QMovie(filename)
        self.tab3.l2.setMovie(gif)
        gif.start()
        #self.tab3.l2.w, self.tab3.l2.h = gif.width(), gif.height()
        #self.tab3.inforEdit1.setText('size of the original gif is ' + str(width) +'*' + str(height)
    
    def showImg(self):
        """显示图片"""
        sender = self.sender()
        if sender == self.tab1.OKButton:
            filename = self.tab1.textEdit.text()
            img = QPixmap(filename)
            self.tab1.l1.setPixmap(img)
            self.tab1.l1.w, self.tab1.l1.h = img.width(), img.height()
            self.tab1.inforEdit.setText('size of the original img is ' + str(img.width()) +'*' + str(img.height()))
        if sender == self.tab2.OKButton:
            filename = self.tab2.textEdit.text()
            img = QPixmap(filename)
            self.tab2.l1.setPixmap(img)
            self.tab2.l1.w, self.tab2.l1.h = img.width(), img.height()
            self.tab2.inforEdit.setText('size of the original img is ' + str(img.width()) +'*' + str(img.height()))
    
    def cleartext(self):
        """消除文本框"""
        sender = self.sender()
        if sender == self.tab1.Cancle:
            self.tab1.textEdit.clear()
        else:
            self.tab2.textEdit.clear()
            
    def cleartext1(self):
        self.tab3.textEdit3.clear()
    #tab2的UI接口 设置雷同tab1
        
    def tab2UI(self):
        """物品移除UI窗口"""
        self.tab2.point = []
        self.tab2.list1=QLineEdit('Points:',self)
        self.tab2.list1.setReadOnly(True)
        self.tab2.inforEdit = QLineEdit()
        self.tab2.inforEdit.setText("Welcome!")
        self.tab2.textEdit = QLineEdit()
        self.tab2.FileButton = QPushButton("Open File")
        self.tab2.OKButton = QPushButton("OK")
        self.tab2.Cancle = QPushButton("Cancle")
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox.addWidget(self.tab2.textEdit)
        hbox.addWidget(self.tab2.FileButton)
        hbox.addWidget(self.tab2.OKButton)
        hbox.addWidget(self.tab2.Cancle)
        hbox.addStretch(1)
        hbox1 = QHBoxLayout()
        self.tab2.functionButton = QPushButton("Object Remove")
        self.tab2.clearButton = QPushButton("clear points")
        hbox1.addWidget(self.tab2.list1)
        hbox1.addWidget(self.tab2.functionButton)
        hbox1.addWidget(self.tab2.clearButton)
        hbox1.addStretch(1)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.tab2.inforEdit)
        hbox2.addStretch(1)
        pixmap = QPixmap("1544672096862777.png")
        #可以进行选点的myLabel类
        self.tab2.l1 = myLabel(self)
        self.tab2.l1.setPixmap(pixmap)
        #self.tab2.l1.setScaledContents(True)
        #self.tab2.l1.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)
        #self.tab2.l1.setFixedSize(self.canvas_w,self.canvas_h)
        vbox.addStretch(1)
        vbox.addWidget(self.tab2.l1)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        self.tab2.setLayout(vbox)
        self.setTabText(1,"image object remove")
        self.tab2.l1.clicked.connect(self.addPoints)
        self.tab2.FileButton.clicked.connect(self.showDialog)
        self.tab2.OKButton.clicked.connect(self.showImg)
        self.tab2.Cancle.clicked.connect(self.cleartext)
        self.tab2.functionButton.clicked.connect(self.object_remove)
        self.tab2.clearButton.clicked.connect(self.clearPoint)
        
        
    def tab3UI(self):
        """视频裁剪UI窗口"""
        self.tab3.inforEdit1 = QLineEdit()
        self.tab3.inforEdit1.setText("Welcome!")
        self.tab3.textEdit3 = QLineEdit()
        self.tab3.FileButton1 = QPushButton("Open File")
        self.tab3.OKButton1 = QPushButton("OK")
        self.tab3.Cancle1 = QPushButton("Cancle")
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox.addWidget(self.tab3.textEdit3)
        hbox.addWidget(self.tab3.FileButton1)
        hbox.addWidget(self.tab3.OKButton1)
        hbox.addWidget(self.tab3.Cancle1)
        hbox.addStretch(1)
        hbox1 = QHBoxLayout()
        self.tab3.textEdit4 = QLineEdit()
        self.tab3.textEdit5 = QLineEdit()
        infor1 = QLabel('Width', self.tab3)
        infor2 = QLabel('Height',self.tab3)
        self.tab3.functionButton1 = QPushButton("Seam")
        hbox1.addWidget(infor1)
        hbox1.addWidget(self.tab3.textEdit4)
        hbox1.addWidget(infor2)
        hbox1.addWidget(self.tab3.textEdit5)
        hbox1.addWidget(self.tab3.functionButton1)
        hbox1.addWidget(self.tab3.inforEdit1)
        hbox1.addStretch(1)
        #
        gif = QMovie("golf.gif")
        self.tab3.l2 = QLabel(self.tab3)
        self.tab3.l2.setMovie(gif)
        gif.start()
        #self.tab3.l2.setScaledContents(True)
        #self.tab3.l2.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)
        
        #self.tab3.l2.setFixedSize(self.canvas_w,self.canvas_h)
        vbox.addStretch(1)
        vbox.addWidget(self.tab3.l2)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox1)
        self.tab3.setLayout(vbox)
        self.tab3.FileButton1.clicked.connect(self.showDialog1)
        self.tab3.OKButton1.clicked.connect(self.showVideo)
        self.tab3.Cancle1.clicked.connect(self.cleartext1)
        self.tab3.functionButton1.clicked.connect(self.seamer_video)
        self.setTabText(2,"video seam carver")
    #退出设置
    def contextMenuEvent(self, event):
        """右键的退出设置"""
        cmenu = QMenu(self)
        quitAct = cmenu.addAction("Quit")
        action = cmenu.exec_(self.mapToGlobal(event.pos()))
        if action == quitAct:
            qApp = QApplication.instance()
            qApp.quit()    
    #tab2中mylabel中通过鼠标点击进行选点
    def addPoints(self):
        """tab2鼠标选点"""
        self.tab2.point.append((round(self.tab2.l1.x),
                            round(self.tab2.l1.y)))
        text = toText(self.tab2.point)
        self.tab2.list1.setText(text)
    
    def clearPoint(self):
        """清除点"""
        self.tab2.point = []
        self.tab2.list1.setText('')

# 可以响应鼠标点击的QLabel类
class myLabel(QLabel):
    """可以响应鼠标点击的QLabel类"""
    clicked = pyqtSignal()

    x,y,w,h=0,0,0,0
    
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.x=e.x()
            self.y=e.y()
            self.clicked.emit()
    
def toText(lis):
    """输出选择点的信息"""
    text = ''
    for i in lis:
        text += '%s,%s\n' % (i[0], i[1])
    return text

if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = TabDemo()
    demo.show()
    sys.exit(app.exec_())