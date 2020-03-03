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
import Plan_Act

class Train:
    def __init__(self):
        self.plan_act = Plan_Act.Plan_Act()
        self.muzero = model_muzero.Muzero(256, 3).compile_model()
        
    def train(self):
        self.plan_act.plan()
        self.plan_act.act()
        print(self.plan_act.replay_buffer)
        for i in range(len(self.plan_act.replay_buffer)):
            self.muzero.fit([self.plan_act.board.board_1, self.plan_act.board.board_2], self.plan_act.replay_buffer[i])
            self.plan_act.board = self.plan_act.board.next(self.plan_act.replay_buffer[i][2]) if not self.plan_act.board.end else self.plan_act.board

train = Train()
train.train()

            