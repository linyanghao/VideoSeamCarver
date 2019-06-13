#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import ig2nx as nx
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter
import collections
from copy import deepcopy
import imageio
import time
import os

OUT_FOLDER = './out'
if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)

def L1Norm(pixel1, pixel2):
    return sum( (I1-I2 if I1>I2 else I2-I1) for I1, I2 in zip(pixel1, pixel2)) 
    
class VideoWrapper():
    def __init__(self, img):
        self.originalPixels = img
        self.augmentedPixels = {}
    def getValue(self, node):
        if isinstance(node[0], int):
            return self.originalPixels[node[0]][node[1]][node[2]]
        else:
            return self.augmentedPixels[node]
    def setValue(self, node, value):
        if isinstance(node[0], int):
            self.originalPixels[node[0]][node[1]][node[2]] = value
        else:
            self.augmentedPixels[node] = value
    
    
class VideoSeamCarver():
    # Node 和 img 有一一对应关系
    # Pos 是删除Seam后Node的位置
    def __init__(self, img, mode='vertical'):
        self.img = VideoWrapper(img)
        self.numFrames = img.shape[0]
        self.numRows = img.shape[1]
        self.numCols = img.shape[2]
        self.Pos2Node = [[[(t, i, j) for j in range(self.numCols)] for i in range(self.numRows)] for t in range(self.numFrames)]
        self.Energy = L1Norm
        self.Node2Img = lambda node: self.img.getValue(node) # returns pixel value
        self.Pos2Img = lambda t, i, j: self.Node2Img(self.Pos2Node[t][i][j]) # returns pixel value
        self._initializeGraph()
        self.PALETTE = {
            'black': np.asarray([0, 0, 0], np.uint8), 
            'red': np.asarray([255, 0, 0], np.uint8)
        }

    def _initializeGraph(self):
        startTime = time.time()
        self.G = nx.DiGraph()
        self.S_Node = 'S'
        self.T_Node = 'T'
        self.G.add_nodes_from([self.S_Node, self.T_Node])
        for j in range(self.numCols):
            for t in range(self.numFrames):
                for i in range(self.numRows):
                    currentNode = self.Pos2Node[t][i][j]
                    self.G.add_node(currentNode)

                    self._addEdges(t, i, j)
        self.G.flush()
        print('Graph Initialization took %s seconds' % (time.time()-startTime))
                
    def _addEdges(self, t, i, j):
        '''
        将Pos为(t, i, j)的Node和与其在Pos意义上相邻的Node，在Graph中连接
        '''
        currentNode = self.Pos2Node[t][i][j]

        if j == 0:
            self.G.add_edge(self.S_Node, currentNode, capacity=float('inf'))
        elif j == len(self.Pos2Node[t][i]) - 1:
            self.G.add_edge(currentNode, self.T_Node, capacity=float('inf'))

        backwardNode = self._getBackwardNode(t, i, j)
        if backwardNode is not None:
            self.G.add_edge(backwardNode, currentNode, capacity=self.Energy(self.Node2Img(backwardNode), self.Node2Img(currentNode)))
            self.G.add_edge(currentNode, backwardNode, capacity=float('inf'))
        
        for diagonalNode in self._getDiagonalNodes(t, i, j):
            self.G.add_edge(currentNode, diagonalNode, capacity=float('inf'))
    def _removeEdges(self, t, i, j):
        currentNode = self.Pos2Node[t][i][j]

        if j == 0:
            self.G.remove_edge(self.S_Node, currentNode)
        elif j == len(self.Pos2Node[t][i]) - 1:
            self.G.remove_edge(currentNode, self.T_Node)

        backwardNode = self._getBackwardNode(t, i, j)
        if backwardNode is not None:
            self.G.remove_edge(backwardNode, currentNode)
            self.G.remove_edge(currentNode, backwardNode)
        
        for diagonalNode in self._getDiagonalNodes(t, i, j):
            self.G.remove_edge(currentNode, diagonalNode)

    def _getBackwardNode(self, t, i, j):
        if j > 0:
            return self.Pos2Node[t][i][j-1]
        else:
            return None
    def _getDiagonalNodes(self, t, i, j):
        result = []
        if j > 0:
            if i > 0: result.append(self.Pos2Node[t][i-1][j-1])
            if i < self.numRows - 1: result.append(self.Pos2Node[t][i+1][j-1])
            if t > 0: result.append(self.Pos2Node[t-1][i][j-1])
            if t < self.numFrames - 1: result.append(self.Pos2Node[t+1][i][j-1])
        return result
    def _getAffectedPoss(self, t, i, j):
        result = []
        if j+1 < len(self.Pos2Node[t][i]):
            result.append((t, i, j+1))
        if j+2 < len(self.Pos2Node[t][i]):
            result.append((t, i, j+2))
        return result

    def _flatten(self, seam):
        '''
        Flattens a 2-D (In frame and row directions) Seam
        '''
        return list(itertools.chain(*seam))

    def Solve(self):
        '''根据当前Graph结构，返回最小割'''
        startTime = time.time()
        cutValue, partition = nx.minimum_cut(self.G, self.S_Node, self.T_Node)
        print('Solving Minimum Cut: %s seconds' % (time.time()-startTime) )
        leftPartition, rightPartition = partition
        leftPartition.remove('S')
        seam = [[-1 for i in range(self.numRows)] for t in range(self.numFrames)]
        '''
        for row, col in leftPartition:
            if col > seam[row]:
                seam[row] = col  # Group By Row, Max of Col
        '''
        for t in range(self.numFrames):
            for i in range(self.numRows):
                for j in range(len(self.Pos2Node[t][i])-1, -1, -1):
                    if self.Pos2Node[t][i][j] in leftPartition:
                        seam[t][i] = (t, i, j)
                        break
        return seam # Contains Positions

    def SolveK(self, k):
        '''根据当前Graph结构，返回前K个最小割'''
        tempG = deepcopy(self.G)
        seams = []
        for _ in range(k):
            startTime = time.time()
            cutValue, partition = nx.minimum_cut(tempG, self.S_Node, self.T_Node)
            leftPartition, rightPartition = partition
            leftPartition.remove('S')
            seam = [[-1 for i in range(self.numRows)] for t in range(self.numFrames)]
            for t in range(self.numFrames):
                for i in range(self.numRows):
                    for j in range(len(self.Pos2Node[t][i])-1, -1, -1):
                        if self.Pos2Node[t][i][j] in leftPartition:
                            tempG.add_edge(self.Pos2Node[t][i][j], self.Pos2Node[t][i][j+1], capacity=float('inf'))
                            tempG.flush()
                            seam[t][i] = (t, i, j)
                            break
            seams.append(seam)
            print('Solving seam %s/%s: %s seconds'% (_+1, k, time.time()-startTime))
        return seams
    
    def RemoveSeam(self, seam):
        '''削除Seam处的一列像素，并对Graph结构进行更新'''
        addEdgesQueue = []
        seam = self._flatten(seam)
        for frame, row, j in seam:
            t = frame
            i = row
            node = self.Pos2Node[frame][row].pop(j)
            self.G.remove_node( node ) # Remove from Pos2Node and G, the seam node in row i. Then Pos2Node[i][j] becomes Pos2Node[i][j+1]
            if j < len(self.Pos2Node[t][i]): # if Pos2Node[i][j+1] not out of bound
                addEdgesQueue.append((t, i, j))
        for t, i, j in addEdgesQueue:
            self._addEdges(t, i, j) 
        self.G.flush()

    def AugmentSeam(self, seam):
        '''在Seam处填充一列像素，并对Graph结构进行更新'''
        insertNodeQueue = []
        addEdgesQueue = []
        seam = self._flatten(seam)
        for frame, row, j in seam:
            t = frame
            i = row
            
            if j+1 < len(self.Pos2Node[t][i]): # if Pos2Node[i][j+1] not out of bound
                augmentation = ( self.Pos2Img(t, i, j)/2 + self.Pos2Img(t, i, j+1)/2 ).astype('uint8') # Averaging
                augmentedNode = (self.Pos2Node[t][i][j], self.Pos2Node[t][i][j+1])
                self.img.setValue(augmentedNode, augmentation)
                self.G.add_node(augmentedNode)
                
                for _t, _i, _j in self._getAffectedPoss(t, i, j):
                    self._removeEdges(_t, _i, _j) 
                insertNodeQueue.append((t, i, j, augmentedNode))
                
            else: # Pos2Node[i][j] connected with T_Node, this should never happen
                raise Exception('??')
        for t, i, j, augmentedNode in insertNodeQueue:
            self.Pos2Node[t][i].insert(j+1, augmentedNode) # CAUTION: Insertion Between j and j+1, Node at j+1 is now augmentedNode
            addEdgesQueue.append((t, i, j+1))
            for _t, _i, _j in self._getAffectedPoss(t, i, j+1):
                addEdgesQueue.append((_t, _i, _j))
        for t, i, j in addEdgesQueue:
            self._addEdges(t, i, j) 
        
        self.G.flush()
            

    def ShowGraph(self):
        raise NotImplementedError()
        '''展示Graph结构'''
        pos = {self.Pos2Node[i][j]:(i, j) for j in range(len(self.Pos2Node[i])) for i in range(self.numRows)}
        pos['S'] = (0, -1)
        pos['T'] = (0, len(self.Pos2Node[0]))
        labels = nx.get_edge_attributes(self.G, 'capacity')
        nx.draw_networkx_edges(self.G, pos=pos)
        nx.draw_networkx_edge_labels(self.G, pos=pos, edge_labels=labels, font_size=8, label_pos=0.3)
        nx.draw_networkx_nodes(self.G, pos=pos)
        nx.draw_networkx_labels(self.G, pos=pos)
        plt.show()

    def GenerateVideoWithSeam(self, seam):
        '''
        根据当前Pos2Node二维数组构造图片

        returns numpy array
        '''
        videoWithSeam = self.GenerateVideo()
        seam = self._flatten(seam) # Flatten the 2-D array
        for t, i, j in seam:
            videoWithSeam[t][i][j] = self.PALETTE['red']
        return videoWithSeam
    def GenerateVideoWithSeams(self, seams):
        '''
        根据当前Pos2Node二维数组构造图片

        returns numpy array
        '''
        videoWithSeams = self.GenerateVideo()
        for seam in seams:
            seam = self._flatten(seam) # Flatten the 2-D array
            for t, i, j in seam:
                videoWithSeams[t][i][j] = self.PALETTE['red']
        return videoWithSeams
    def GenerateVideo(self):
        '''
        根据当前Pos2Node二维数组构造图片
        
        returns numpy array
        '''
        video = np.asarray([[
                            [self.Node2Img(node)
                            for node in row]
                        for row in frame]
                        for frame in self.Pos2Node])
        return video
    def ShowImg(self, seam, numCols):
        raise NotImplementedError()
        imgWithSeam = self.GenerateImgWithSeam(seam, numCols)

        plt.imshow(imgWithSeam)
        plt.show()

if __name__ == '__main__':
    IMAGE_AS_VIDEO, SMALL_DATA, REMOVE_SEAM_TEST, AUGMENT_SEAM_TEST = True, False, False, True
    REMOVE_SEAMS_COUNT = 40
    AUGMENT_SEAMS_COUNT = 10
    
    if IMAGE_AS_VIDEO:
        img = Image.open('2.png')
        img = img.convert('RGB')
        img = np.array(img)
        video = np.reshape(img, [1, *img.shape])
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

    if REMOVE_SEAM_TEST: # 测试减少图片宽度的功能
        videos = []
        for i in range(REMOVE_SEAMS_COUNT):
            startTime = time.time()
            print(i)
            seam = carver.Solve()
            #carver.ShowImg(seam)
            video = carver.GenerateVideoWithSeam(seam)
            imageio.mimsave(OUT_FOLDER+'/%s.gif'% i, video)
            videos.append(video)
            carver.RemoveSeam(seam)
            print('Total Time Removing Seam %s/%s: %s seconds'% (i, REMOVE_SEAMS_COUNT, time.time()-startTime))
        video = carver.GenerateVideo()
        imageio.mimsave(OUT_FOLDER+'/result.gif', video)
        
        #carver.ShowImg(None)

    if AUGMENT_SEAM_TEST: # 测试增加图片宽度的功能
        seams = carver.SolveK(AUGMENT_SEAMS_COUNT)
        videoWithSeams = carver.GenerateVideoWithSeams(seams)
        imageio.mimsave(OUT_FOLDER+'/videoWithSeams.gif', videoWithSeams)

        for seam in seams:
            carver.AugmentSeam(seam)
        videoAugmented = carver.GenerateVideo()
        
        imageio.mimsave(OUT_FOLDER+'/videoAugmented.gif', videoAugmented)
