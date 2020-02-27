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


class Residual_Block(Model):
    def __init__(self, filters_num: int, dropout_rate: float = 0.1, stride: int = 1, *args, **kwargs):
        super(Residual_Block, self).__init__(*args, **kwargs)
        self.filters_num = filters_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        
        self.conv = Conv2D(filters=self.filters_num,kernel_size=1, strides=self.stride, padding='same')
        self.conv_1 = Conv2D(filters=self.filters_num,kernel_size=3, strides=self.stride, padding='same')
        self.conv_2 = Conv2D(filters=self.filters_num * 4, kernel_size=1, strides=self.stride, padding='same')
        self.conv_3 = Conv2D(filters=self.filters_num * 4, kernel_size=1, strides=self.stride, padding='same')
        self.batch_norm = BatchNormalization()
        self.batch_norm_1 = BatchNormalization()
        self.batch_norm_2 = BatchNormalization()
        self.relu = ReLU()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.conv_3.trainable = False
    
    def call(self, inputs):
        batch_normed_1 = self.batch_norm(inputs)
        relued_1 = self.relu(batch_normed_1)
        conv_1 = self.conv(relued_1)
        batch_normed_2 = self.batch_norm_1(conv_1)
        relued_2 = self.relu(batch_normed_2)
        conv_2 = self.conv_1(relued_2)
        batch_normed_3 = self.batch_norm_2(conv_2)
        relued_3 = self.relu(batch_normed_3)
        conv_3 = self.conv_2(relued_3)
        #inputのfilter数が足りない時はself.conv_3を使う
        return inputs + conv_3

class MuZero_Model_pred(Model):
    def __init__(self, *args, **kwargs):
        super(MuZero_Model_pred, self).__init__(*args, **kwargs)
        filters_num = 128
        stride = 2
        kernel_size = 3
        padding = 'valid'
        self.conv = Conv2D(filters=filters_num, kernel_size=kernel_size, strides=stride, padding=padding)
        self.resblock_nums = [2, 3, 3]
        self.resblocks = []
        self.resblocks += [[Residual_Block(filters_num, stride=stride) for _ in range(self.resblock_nums[0])]]
        filters_num *= 2
        self.conv_1 = Conv2D(filters=filters_num, kernel_size=kernel_size, strides=stride, padding=padding)
        self.resblocks += [[Residual_Block(filters_num, stride=stride) for _ in range(self.resblock_nums[1])]]
        self.avgpool = AveragePooling2D(strides=stride)
        self.resblocks += [[Residual_Block(filters_num, stride=stride) for _ in range(self.resblock_nums[2])]]
        self.avgpool_1 = AveragePooling2D(strides=stride)
    
    def forward_resblock(self, layers, inputs):
        for layer in layers:
            inputs = layer(inputs)
        return inputs
    
    def call(self, inputs):
        outputs = self.forward_resblock(self.resblocks[0], self.conv(inputs))
        outputs = self.forward_resblock(self.resblocks[1], self.conv_1(outputs))
        outputs = self.forward_resblock(self.resblocks[2], self.avgpool(outputs))
        return self.avgpool_1(outputs)


class MuZero_Model_dynamics(Model):
    def __init__(self, *args, **kwargs):
        super(MuZero_Model_dynamics, self).__init__(*args, **kwargs)
        filters_num = 128
        stride = 2
        kernel_size = 3
        padding = 'valid'
        self.conv = Conv2D(filters=filters_num, kernel_size=kernel_size, strides=stride, padding=padding)
        self.resblock_nums = [3, 3, 3]
        self.resblocks = []
        self.resblocks += [[Residual_Block(filters_num, stride=stride) for _ in range(self.resblock_nums[0])]]
        filters_num *= 2
        self.conv_1 = Conv2D(filters=filters_num, kernel_size=kernel_size, strides=stride, padding=padding)
        self.resblocks += [[Residual_Block(filters_num, stride=stride) for _ in range(self.resblock_nums[1])]]
        self.avgpool = AveragePooling2D(strides=stride)
        self.resblocks += [[Residual_Block(filters_num, stride=stride) for _ in range(self.resblock_nums[2])]]
        self.avgpool_1 = AveragePooling2D(strides=stride)
        self.flat = Flatten()
        self.dense = Dense(1, activation='sigmoid')
        
    
    def forward_resblock(self, layers, inputs):
        for layer in layers:
            inputs = layer(inputs)
        return inputs
    
    def call(self, inputs):
        outputs = self.forward_resblock(self.resblocks[0], self.conv(inputs))
        outputs = self.forward_resblock(self.resblocks[1], self.conv_1(outputs))
        outputs = self.forward_resblock(self.resblocks[2], self.avgpool(outputs))
        return self.avgpool_1(outputs)

class MuZero:
    def __init__(self, img_dim: int, img_channel: int):
        self.img_dim = img_dim
        self.img_channel = img_channel
        self.input_layer = Input(shape=(img_dim, img_dim, img_channel))
        self.muzero_pred = MuZero_Model_pred()
        self.muzero_dynamics = MuZero_Model_dynamics()
        
    
    def compile_model(self):
        outputs = self.muzero_pred(self.input_layer)
        model_pred = Model(inputs=[self.input_layer], outputs=[outputs])
        model_pred.summary()
        model_pred.compile(
            optimizer=tfk.optimizers.Adam(lr=0.0002),
        )

        
        
        
        

        

        