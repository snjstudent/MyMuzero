from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, BatchNormalization, ReLU, Dropout, MaxPool2D, AveragePooling2D,Softmax
from tensorflow.keras.models import Model,Sequential
import tensorflow_addons as tfa
import numpy as np
import random
import sys
import time
from operator import attrgetter
import math
import itertools
from datetime import datetime
import pickle

class GameBoard:
    def __init__(self,
    board_1=[[0 for i in range(9)] for u in range(9)],
    board_2=[[0 for i in range(9)] for u in range(9)]):
        self.board_1 = board_1
        self.board_2 = board_2
        self.count = 0
        self.action_count = 0


    @property
    def lose(self):
        board_tmp = np.array(self.board_1).reshape(9, 9)
        for i in range(9):
            for u in range(9):
                self.depth_first_search(i, u, board_tmp)
                if self.count >= 5 or (self.board_1[i][u] * self.board_2[i][u]):
                    return True
                self.count = 0
        return False
    
    @property
    def end(self):
        return self.lose or self.draw
    
    @property
    def draw(self):
        return self.action_count == 81

    @property
    def legal_actions(self):
        tmp = []
        for i in range(len(self.board_1)):
            for u in range(len(self.board_1[i])):
                if self.board_1[i][u] == 0 and self.board_2[i][u] == 0:
                    tmp.append((i + 1) * (u * 1))
                else:
                    tmp.append(0)
        return tuple(tmp)

    def next(self, action):
        q, mod = divmod(len(self.board_1[0]), len(self.board_1[0]))
        self.board_1[q - 1 if q > 0 else q][mod - 1 if mod > 0 else mod] = 1
        return GameBoard(self.board_2, self.board_1)

    def depth_first_search(self, x, y, board):
        board[x][y] = 0
        for dx in range(-1, 1):
            for dy in range(-1, 1):
                nx, ny = x + dx, y + dy
                if (nx >= 0 and nx <= 9 and ny >= 0 and ny <= 9 and board[nx][ny] > 0):
                    self.depth_first_search(nx, ny, board)
                    count += 1
        return
  