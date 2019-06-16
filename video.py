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

images = []

OUT_FOLDER = './out'
if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)

def L1Norm(pixel1, pixel2):
    return sum((I1-I2 if I1>I2 else I2-I1) for I1, I2 in zip(pixel1, pixel2)) 
    
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
    # Pos是删除Seam后Node的位置
    def __init__(self, img, mode='vertical'):
        self.Node2Pixel       = VideoWrapper(img)
        self.numFrames = img.shape[0]
        self.numRows   = img.shape[1]
        self.numCols   = img.shape[2]
        self.Pos2Node  = [[[(t, i, j) for j in range(self.numCols)]     \
                                        for i in range(self.numRows)]   \
                                        for t in range(self.numFrames)]
        self.Node2Img  = lambda node: self.Node2Pixel.getValue(node) # returns pixel value
        self.Pos2Img   = lambda t, i, j: self.Node2Img(self.Pos2Node[t][i][j]) # returns pixel value
        self._initializeGraph()
        self.PALETTE = {
            'black': np.asarray([0, 0, 0], np.uint8),
            'red'  : np.asarray([255, 0, 0], np.uint8)
        }

    def _initializeGraph(self):
        startTime = time.time()

        self.G      = nx.DiGraph()
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
            self.G.add_edge(backwardNode, currentNode, 
                    capacity=self._energy(t, i, j-1)
            )
            self.G.add_edge(currentNode, backwardNode, capacity=float('inf'))
        '''
        upwardNode = self._getUpwardNode(t, i, j) # 注意！这条边在原论文是没有的！
        if upwardNode is not None:
            energy = L1Norm(self.Node2Img(upwardNode), self.Node2Img(currentNode))
            self.G.add_edge(upwardNode, currentNode, 
                    capacity=-energy
            )
            self.G.add_edge(currentNode, upwardNode, 
                    capacity=energy
            )
        '''
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
        '''
        upwardNode = self._getUpwardNode(t, i, j) # 注意！这条边在原论文是没有的！
        if upwardNode is not None:
            self.G.remove_edge(upwardNode, currentNode)
            self.G.remove_edge(currentNode, upwardNode)
        '''
        for diagonalNode in self._getDiagonalNodes(t, i, j):
            self.G.remove_edge(currentNode, diagonalNode)

    def _getUpwardNode(self, t, i, j):
        if i > 0:
            return self.Pos2Node[t][i-1][j]
        else:
            return None

    def _getBackwardNode(self, t, i, j):
        if j > 0:
            return self.Pos2Node[t][i][j-1]
        else:
            return None

    def _getDiagonalNodes(self, t, i, j):
        result = []
        if j > 0:
            if i > 0                 : result.append(self.Pos2Node[t][i-1][j-1])
            if i < self.numRows - 1  : result.append(self.Pos2Node[t][i+1][j-1])
            if t > 0                 : result.append(self.Pos2Node[t-1][i][j-1])
            if t < self.numFrames - 1: result.append(self.Pos2Node[t+1][i][j-1])
        return result
    

    def _getAffectedPoss_beforeInsertion(self, t, i, j):
        result = []
        result.append((t, i, j)) #这是加入垂直方向的边以后才需要的
        if j+1 < len(self.Pos2Node[t][i]):
            result.append((t, i, j+1))
        if j+2 < len(self.Pos2Node[t][i]):
            result.append((t, i, j+2))
        return result
    
    def _getAffectedPoss_afterInsertion(self, t, i, j):
        result = []
        if j-1 >= 0: #这是加入垂直方向的边以后才需要的
            result.append((t, i, j-1))
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

    def _energy(self, t, i, j): 
        left  = j-1 if j-1 >= 0 else j
        right = j+1 if j+1  <len(self.Pos2Node[t][i]) else j
        up    = i-1 if i-1 >= 0 else i
        down  = i+1 if i+1 < len(self.Pos2Node[t]) else i
        energy = L1Norm(self.Pos2Img(t, i, left), self.Pos2Img(t, i, right)) / (right-left) + \
                 L1Norm(self.Pos2Img(t, up, j), self.Pos2Img(t, down, j)) / (down-up)
        
        return energy

    def Solve(self):
        '''根据当前Graph结构，返回最小割'''
        startTime = time.time()
        cutValue, partition = nx.minimum_cut(self.G, self.S_Node, self.T_Node)
        print('Solving Minimum Cut: %s seconds' % (time.time()-startTime) )
        print('CutValue: %s' % (cutValue))
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
                            tempG.add_edge(self.Pos2Node[t][i][j], self.Pos2Node[t][i][j+1], 
                                           capacity=float('inf'))
                            tempG.flush()
                            seam[t][i] = (t, i, j)
                            break
            seams.append(seam)
            print('Solving seam %s/%s: %s seconds'% (_+1, k, time.time()-startTime))
        return sorted(seams, key=lambda seam:seam[0][0][2], reverse=True) # t=0, i=0 处，j最大的优先
    
    def RemoveSeam(self, seam):
        '''削除Seam处的一列像素，并对Graph结构进行更新'''
        removeNodeQueue = []
        addEdgesQueue = []
        seam = self._flatten(seam)
        for frame, row, j in seam:
            t = frame
            i = row

            for _t, _i, _j in self._getAffectedPoss_afterInsertion(t, i, j):
                self._removeEdges(_t, _i, _j) 
            removeNodeQueue.append((t, i, j))
        
        for t, i, j in removeNodeQueue:
            node = self.Pos2Node[t][i].pop(j)
            for _t, _i, _j in self._getAffectedPoss_beforeInsertion(t, i, j-1):
                addEdgesQueue.append((_t, _i, _j))

            # Remove from Pos2Node and G, the seam node in row i. Then Pos2Node[i][j] becomes Pos2Node[i][j+1]
            self.G.remove_node( node ) 

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
                augmentation  = (self.Pos2Img(t, i, j)/2 + self.Pos2Img(t, i, j+1)/2 ).astype('uint8') # Averaging
                augmentedNode = (self.Pos2Node[t][i][j], self.Pos2Node[t][i][j+1])
                self.Node2Pixel.setValue(augmentedNode, augmentation)
                self.G.add_node(augmentedNode)
                
                for _t, _i, _j in self._getAffectedPoss_beforeInsertion(t, i, j):
                    self._removeEdges(_t, _i, _j) 
                insertNodeQueue.append((t, i, j, augmentedNode))
                
            else: # Pos2Node[i][j] connected with T_Node, this should never happen
                raise Exception('??')
        for t, i, j, augmentedNode in insertNodeQueue:
            self.Pos2Node[t][i].insert(j+1, augmentedNode) # CAUTION: Insertion Between j and j+1, Node at j+1 is now augmentedNode
            addEdgesQueue.append((t, i, j+1))
            for _t, _i, _j in self._getAffectedPoss_afterInsertion(t, i, j+1):
                addEdgesQueue.append((_t, _i, _j))
        for t, i, j in addEdgesQueue:
            self._addEdges(t, i, j) 
        
        self.G.flush()

    def ShowGraph(self):
        raise NotImplementedError()
        '''展示Graph结构'''
        pos = {self.Pos2Node[i][j]:(i, j) for j in range(len(self.Pos2Node[i]))  \
                                           for i in range(self.numRows)          \
                                                }
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
        seam          = self._flatten(seam) # Flatten the 2-D array
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
        video = np.asarray([[[self.Node2Img(node)
                            for node in row]
                        for row in frame]
                        for frame in self.Pos2Node])
        return video

    def ShowImg(self, seam, numCols):
        raise NotImplementedError()
        imgWithSeam = self.GenerateImgWithSeam(seam, numCols)

        plt.imshow(imgWithSeam)
        plt.show()

    def shrink_hor(self, rm_count, save_hist=False, want_trans=False):
        # videos = []
        for i in range(rm_count):
            # print(i)
            startTime = time.time()
            seam      = self.Solve()    

            # Image with removed seam highlighted
            if save_hist:
                video_w_seam = self.GenerateVideoWithSeam(seam)
                imageio.mimsave(OUT_FOLDER+'/%s.gif' % i, video_w_seam if not want_trans else self.trans(video_w_seam))
                # videos.append(video_w_seam)
            
            self.RemoveSeam(seam)
            # print('Total Time Removing Seam %s/%s: %s seconds' % (i, rm_count, time.time()-startTime))

        videoAugmented = self.GenerateVideo()
        # Save the final image
        imageio.mimsave(OUT_FOLDER+'/result.gif', videoAugmented)
        return videoAugmented

    # Horizontal augmentation
    def augment_hor(self, aug_count, save_hist=True, want_trans=False):

        seams = self.SolveK(aug_count)

        if save_hist:
            videoWithSeams = self.GenerateVideoWithSeams(seams)
            imageio.mimsave(OUT_FOLDER+'/videoWithSeams.gif', videoWithSeams if not want_trans else self.trans(videoWithSeams))

        for seam in seams:
            self.AugmentSeam(seam)

        videoAugmented = self.GenerateVideo()
        imageio.mimsave(OUT_FOLDER+'/videoAugmented.gif', videoAugmented)
        return videoAugmented
        
    def trans(self, video):
        return np.transpose(video, (0,2,1,3))

    def scale_hor(self, pix_count, save_hist=False):
        if pix_count == 0:
            return self.GenerateVideo()
        
        if pix_count < 0:
            return self.shrink_hor(-pix_count, save_hist)
        elif pix_count > 0:
            return self.augment_hor(pix_count, save_hist)
    
    

if __name__ == '__main__':
    IMAGE_AS_VIDEO, SMALL_DATA = True, False

    REMOVE_SEAM_TEST  = False
    AUGMENT_SEAM_TEST = True
    XY_SCALE_TEST     = False

    assert XY_SCALE_TEST + REMOVE_SEAM_TEST + AUGMENT_SEAM_TEST == 1, "Wrong setting!"

    REMOVE_SEAMS_COUNT  = 40
    AUGMENT_SEAMS_COUNT = 40
    X_SEAMS_COUNT       = 60
    Y_SEAMS_COUNT       = 60
    
    if IMAGE_AS_VIDEO:
        img   = Image.open('dolphin.png')
        img   = img.convert('RGB')
        img   = np.array(img)
        video = np.reshape(img, [1, *img.shape])
        print(video.shape)
    else:
        video = imageio.get_reader('waterski_low_resolution.mov', 'ffmpeg')
        frames = []
        for i, frame in enumerate(video):
            frames.append(frame)
        video = np.array(frames)

    if SMALL_DATA:
        video = video[:3, ...]
    carver = VideoSeamCarver(video)
    #carver.Draw()

    if REMOVE_SEAM_TEST: # 测试减少图片宽度的功能
        print("=========== REMOVE_SEAM_TEST ==============\n")
        res = carver.shrink_hor(REMOVE_SEAMS_COUNT, save_hist=True)
        assert res.shape[2] == video.shape[2] - REMOVE_SEAMS_COUNT, "{} is not {}".format(

           res.shape[2], video.shape[2] - REMOVE_SEAMS_COUNT
        )

        images = []
        for i in range(1,REMOVE_SEAMS_COUNT):
            filename = OUT_FOLDER + "/{}.gif".format(i)
            images.append(imageio.imread(filename))
        imageio.mimsave(OUT_FOLDER + "/movie.gif", images)

    if AUGMENT_SEAM_TEST: # 测试增加图片宽度的功能
        print("=========== AUGMENT_SEAM_TEST ==============\n")
        frames = []
        seams = carver.SolveK(AUGMENT_SEAMS_COUNT)
        video_w_seams = carver.GenerateVideoWithSeams(seams)
        imageio.mimsave(OUT_FOLDER + "/dolphin_seams.gif", video_w_seams)
        for seam in seams:
            video_w_seam = carver.GenerateVideoWithSeam(seam)
            frames.append(video_w_seam[0])
            carver.AugmentSeam(seam)
        blank = np.asarray([[carver.PALETTE['black'] for j in range(AUGMENT_SEAMS_COUNT)] for i in range(carver.numRows)])
        print(blank.shape)
        print(frames[0].shape)
        frames[0] = np.concatenate([frames[0], blank], axis=1)
        imageio.mimsave(OUT_FOLDER + "/dolphin_augment_movie.gif", frames)
        assert res.shape[2] == video.shape[2] + AUGMENT_SEAMS_COUNT, "{} is not {}".format(
            res.shape[2], video.shape[2] + AUGMENT_SEAMS_COUNT
        )

    if XY_SCALE_TEST:
        print("=========== XY_SCALE_TEST ==============\n")
        print("=========== {} =========".format(
            "Shrink X" if X_SEAMS_COUNT < 0 else "Augment X"
        ))
        x_scaled_video = carver.scale_hor(X_SEAMS_COUNT, save_hist=True)
        x_scaled_trans = np.transpose(x_scaled_video, (0,2,1,3))

        print("=========== {} =========".format(
            "Shrink Y" if Y_SEAMS_COUNT < 0 else "Augment Y"
        ))
        y_scaler = VideoSeamCarver(x_scaled_trans)
        res = y_scaler.scale_hor(Y_SEAMS_COUNT, save_hist=True)
        res = np.transpose(res, (0,2,1,3))
