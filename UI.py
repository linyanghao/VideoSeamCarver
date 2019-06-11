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
class TabDemo(QTabWidget):
    
    def __init__(self,parent=None):
        super(TabDemo,self).__init__(parent)
        width = QDesktopWidget().availableGeometry().width() * 3 / 4
        height = width/2
        self.canvas_w = width*2/5
        self.canvas_h = width*2/5

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
    
    def tab1UI(self):
        self.tab1.inforEdit = QLineEdit()
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
        pixmap = QPixmap(r"C:\Users\JUNJIN\Pictures\6.jpg")
        self.tab1.l1 = QLabel(self.tab1)
        self.tab1.l1.setPixmap(pixmap)
        self.tab1.l1.setScaledContents(True)
        self.tab1.l1.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)
        self.tab1.l1.setFixedSize(self.canvas_w,self.canvas_h)
        vbox.addStretch(1)
        vbox.addWidget(self.tab1.l1)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox1)
        self.tab1.setLayout(vbox)
        self.setTabText(0,"image seam carver")
        self.tab1.FileButton.clicked.connect(self.showDialog)
        self.tab1.OKButton.clicked.connect(self.showImg)
        self.tab1.Cancle.clicked.connect(self.cleartext)
        self.tab1.functionButton.clicked.connect(self.seamer)
    
    def object_remove(self):
        """
        目标移除函数
        图片文件在filename = self.tab2.textEdit.text()
        选取的点数据在self.tab2.point = []
        将函数放在这里，跑出结果
        保存结果在本地
        然后在self.tab2.l1中以动图形式显示
        """
        pass
    
    def seamer(self):
        """
        seamer函数
        图片文件在filename = self.tab1.textEdit.text()
        将函数放在这里，跑出结果
        保存结果在本地
        然后在self.tab1.l1中显示以动图形式显示
        """
        pass
    
    def seamer_video(self):
        """
        seamer函数对video的处理（应该是和seamer调用相同的外部函数）
        图片文件在filename = self.tab3.textEdit.text()
        将函数放在这里，跑出结果
        保存结果在本地
        然后在self.tab3.l2中显示以动图形式显示
        """
        pass
        
    def showDialog(self):
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
        filename,  _ = QFileDialog.getOpenFileName(self, 'Open file', './')
        
        if filename[-3:] == "gif" or filename[-3:] == "mov":
            self.tab3.textEdit3.setText(filename)
        else:
            pass
        
    def showVideo(self):
        filename = self.tab3.textEdit3.text()
        gif = QMovie(filename)
        self.tab3.l2.setMovie(gif)
        gif.start()
        #self.tab3.l2.w, self.tab3.l2.h = gif.width(), gif.height()
        #self.tab3.inforEdit1.setText('size of the original gif is ' + str(width) +'*' + str(height)
    
    def showImg(self):

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
        sender = self.sender()
        if sender == self.tab1.Cancle:
            self.tab1.textEdit.clear()
        else:
            self.tab2.textEdit.clear()
            
    def cleartext1(self):
        self.tab3.textEdit3.clear()
        
    def tab2UI(self):
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
        pixmap = QPixmap(r"C:\Users\JUNJIN\Pictures\6.jpg")
        self.tab2.l1 = myLabel(self)
        self.tab2.l1.setPixmap(pixmap)
        self.tab2.l1.setScaledContents(True)
        self.tab2.l1.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)
        self.tab2.l1.setFixedSize(self.canvas_w,self.canvas_h)
        vbox.addStretch(1)
        vbox.addWidget(self.tab2.l1)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox1)
        self.tab2.setLayout(vbox)
        self.setTabText(1,"image object remove")
        self.tab2.l1.clicked.connect(self.addPoints)
        self.tab2.FileButton.clicked.connect(self.showDialog)
        self.tab2.OKButton.clicked.connect(self.showImg)
        self.tab2.Cancle.clicked.connect(self.cleartext)
        self.tab2.functionButton.clicked.connect(self.object_remove)
        self.tab2.clearButton.clicked.connect(self.clearPoint)
        
        
    def tab3UI(self):
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
        gif = QMovie(r"C:\Users\JUNJIN\Desktop\1.gif")
        self.tab3.l2 = QLabel(self.tab3)
        self.tab3.l2.setMovie(gif)
        gif.start()
        self.tab3.l2.setScaledContents(True)
        self.tab3.l2.setSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum)
        
        self.tab3.l2.setFixedSize(self.canvas_w,self.canvas_h)
        vbox.addStretch(1)
        vbox.addWidget(self.tab3.l2)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox1)
        self.tab3.setLayout(vbox)
        self.tab3.FileButton1.clicked.connect(self.showDialog1)
        self.tab3.OKButton1.clicked.connect(self.showVideo)
        self.tab3.Cancle1.clicked.connect(self.cleartext1)
        self.tab3.functionButton1.clicked.connect(self.seamer)
        self.setTabText(2,"video seam carver")
    
    def contextMenuEvent(self, event):
        cmenu = QMenu(self)
        quitAct = cmenu.addAction("Quit")
        action = cmenu.exec_(self.mapToGlobal(event.pos()))
        if action == quitAct:
            qApp = QApplication.instance()
            qApp.quit()    
    def addPoints(self):
        self.tab2.point.append((round(self.tab2.l1.x/self.canvas_w*self.tab2.l1.w),
                            round(self.tab2.l1.y/self.canvas_h*self.tab2.l1.h)))
        text = toText(self.tab2.point)
        self.tab2.list1.setText(text)
    
    def clearPoint(self):
        self.tab2.point = []
        self.tab2.list1.setText('')

# 可以响应鼠标点击的QLabel类
class myLabel(QLabel):

    clicked = pyqtSignal()

    x,y,w,h=0,0,0,0
    
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.x=e.x()
            self.y=e.y()
            self.clicked.emit()
    
def toText(lis):
    text = ''
    for i in lis:
        text += '%s,%s\n' % (i[0], i[1])
    return text

if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = TabDemo()
    demo.show()
    sys.exit(app.exec_())