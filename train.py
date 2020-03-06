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
        self.muzero_model = model_muzero.Muzero(9, 2)
        self.muzero = self.muzero_model.compile_model()
        self.board = GameBoard_tmp.GameBoard()

    def copy_layer_weights(self, model_train, model_mcts):
        #trainで訓練されたモデルをモンテカルロ用のモデルにコピーする
        weight_h, weight_f, weight_g = model_train.get_layer_weight()
        model_mcts.h.set_weights(weight_h)
        model_mcts.f.set_weights(weight_f)
        model_mcts.g.set_weights(weight_g)

    def save_weights(self, model):
        #重みを保存する
        pass

    def save_train_datas(self, train_datas):
        #一応訓練データも保存できるようにしておく
        pass

    def load_weights(self, model_train, model_h, model_mct):
        #訓練再開時にモデルの重さを読み込む
        pass

    def make_train_data(self, buffers):
        actions = [i[-1][0].tolist() for i in buffers]
        buffers = [i[:-1] for i in buffers]
        train_datas = []
        for buffer in buffers:
            for i, data in enumerate(buffer):
                train_datas.append(np.array([data],dtype=np.float32)) if i != 1 else train_datas.append(np.array([data[0][0]],dtype=np.float32))
        """
        for train_data in train_datas:
            train_data = np.array(train_data, dtype=np.float32)
        """
        actions = np.array([actions], dtype=np.float32).reshape([1, 9, 9, 5])
        return actions, train_datas
        
    def train(self):
        self.plan_act.plan()
        self.plan_act.act()
        for i in range(len(self.plan_act.replay_buffer) - 5):
            train_buffer = self.plan_act.replay_buffer[i:i + 5]
            action, train_datas = self.make_train_data(train_buffer)
            try:
              self.muzero.fit(np.concatenate([np.array([[[self.board.board_1, self.board.board_2]]], dtype=np.float32).reshape([1, 9, 9, 2]), action], axis=-1), \
              train_datas, batch_size=1)
            except Exception as e:
              print(e)
            self.board = self.board.next(train_buffer[0][3]) if not self.board.end else self.board
        self.copy_layer_weights(self.muzero_model, self.plan_act.model_mcts)

if __name__ == "__main__":
    train = Train()
    train.train()


