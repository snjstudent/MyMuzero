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
import GameBoard

STEPS_PER_MOVE = 300

class Plan_Act:
    def __init__(self):
        self.board = GameBoard()
        self.model_h, MCT.model = model_muzero.MuZero_MCTS(9, 2).compile_model()
        self.trainmodel = model_muzero.Muzero(256, 256, 3).compile_model()
        self.node = MCT.node(self.model_h(self.board.board_1), 1, 1)
        self.replay_buffer = []

    def plan(self):
        self.node.expansion()

    def act(self):
        replay_buffer_tmp = []
        for _ in range(STEPS_PER_MOVE):
            self.node.backup()
        replay_buffer_tmp += [self.node.policy, self.node.mean_value, self.node.reward]
        next_action = self.node.selection().action
        replay_buffer_tmp += [next_action]
        self.replay_buffer.append(replay_buffer_tmp)
        
        

        
        
        
