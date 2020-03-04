from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, ReLU, LeakyReLU, Dropout, AveragePooling2D, LayerNormalization,BatchNormalization
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model,Sequential
#import tensorflow_addons as tfa
import numpy as np
import random
import sys
import MCT
import model_muzero
import GameBoard_tmp

STEPS_PER_MOVE = 300

class Plan_Act:
    def __init__(self):
        self.board = GameBoard_tmp.GameBoard()
        self.model_h, MCT.model = model_muzero.MuZero_MCTS(9, 2).compile_model()
        #self.trainmodel = model_muzero.Muzero(9, 2).compile_model()
        self.node = MCT.node(self.model_h(np.array([[self.board.board_1, self.board.board_2]], dtype=np.float32).reshape([1,9,9,2]))[0], 0, 0)
        self.replay_buffer = []

    def plan(self):
        self.node = MCT.node(self.model_h(np.array([[self.board.board_1, self.board.board_2]], dtype=np.float32).reshape([1,9,9,2]))[0], 0, 0)
        self.node.expansion()
    
    def act(self):
        #本当はゲーム終了まで続けるのでwhile(True)
        for i in range(6):
            replay_buffer_tmp = []
            for _ in range(STEPS_PER_MOVE):
                self.node.backup()
            replay_buffer_tmp += [self.node.policy, self.node.mean_value, self.node.reward]
            replay_buffer_tmp = [np.array(i, dtype=np.float32) for i in replay_buffer_tmp]
            next_action = self.node.selection().action
            replay_buffer_tmp += [np.array([next_action], dtype=np.float32)]
            self.replay_buffer.append(replay_buffer_tmp)
            if not self.board.end:
                self.board.next(next_action)
        #rewardは決着がつく（エピソード終了時）までわからないので
        #最終時間に、勝った、負けたに応じてrewardを付与する
        for i, buffer in enumerate(self.replay_buffer):
            buffer[2] = -1 if self.board.lose and i % 2 == 0 else 0.5 if self.board.draw else 1 
        return
