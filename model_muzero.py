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
        self.dense = Dense(self.filters_num * 4)
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
        return self.dense(inputs) + conv_3

#16層、predict 
#20層、dynamics、presentation
class MuZero_Model(Model):
    def __init__(self,resblock_nums, *args, **kwargs):
        super(MuZero_Model, self).__init__(*args, **kwargs)
        filters_num = 128
        #本当はストライド2、２だとaveragepoolingができないので１に
        stride = 1
        kernel_size = 3
        padding = 'same'
        self.conv = Conv2D(filters=filters_num, kernel_size=kernel_size, strides=stride, padding=padding)
        self.resblock_nums = resblock_nums
        self.resblocks = []
        self.resblocks += [[Residual_Block(int(filters_num / 4), stride=stride) for _ in range(self.resblock_nums[0])]]
        filters_num *= 2
        self.conv_1 = Conv2D(filters=filters_num, kernel_size=kernel_size, strides=stride, padding=padding)
        self.resblocks += [[Residual_Block(int(filters_num / 8), stride=stride) for _ in range(self.resblock_nums[1])]]
        self.avgpool = AveragePooling2D(strides=stride, padding=padding)
        self.resblocks += [[Residual_Block(int(filters_num / 8), stride=stride) for _ in range(self.resblock_nums[2])]]
        self.avgpool_1 = AveragePooling2D(strides=stride, padding=padding)
        self.flat = Flatten()
        self.dense = Dense(1, activation='relu')
        self.dense_1 = Dense(1, activation='softmax')
    
    def forward_resblock(self, layers, inputs):
        for layer in layers:
            inputs = layer(inputs)
        return inputs
    
    def call(self, inputs):
        outputs = self.forward_resblock(self.resblocks[0], self.conv(inputs))
        outputs = self.forward_resblock(self.resblocks[1], self.conv_1(outputs))
        outputs = self.forward_resblock(self.resblocks[2], self.avgpool(outputs))
        flat = self.flat(outputs)
        self.densed = self.dense(flat)
        self.densed_1 = self.dense_1(flat)
        return self.avgpool_1(outputs) if 1 not in outputs.shape[1:-1] else outputs

class MuZero_MCTS:
    def __init__(self, img_dim: int, img_channel: int):
        self.inputs = Input(shape=(img_dim, img_dim, img_channel))
        self.inputs_2 = Input(shape=(img_dim, img_dim, 128 + 1))
        self.inputs_1 = Input(shape=(img_dim, img_dim, 128))
        self.h = MuZero_Model([3, 3, 3])
        self.g = MuZero_Model([3, 3, 3])
        self.f = MuZero_Model([2, 3, 3])
        input_shape = tf.TensorShape([None, img_dim, img_dim, img_channel])
        self.h.build(input_shape)
        input_shape = tf.TensorShape([None, img_dim, img_dim, 128 + 1])
        self.g.build(input_shape)
        input_shape = tf.TensorShape([None, img_dim, img_dim, 128])
        self.f.build(input_shape)
    
    def compile_model(self):
        out_h = self.h(self.inputs)
        out_g = self.g(self.inputs_2)
        out_f = self.f(self.inputs_1)
        model_h = Model(inputs=[self.inputs], outputs=[out_h, self.h.densed])
        model_fg = Model(inputs=[self.inputs_1, self.inputs_2], outputs=[out_g, self.g.densed, out_f, self.f.densed])
        
        '''

        BCEに直しておく！

        '''
        
        model_h.compile(optimizer=tfk.optimizers.Adam(lr=0.0002),
        loss=['mean_squared_error','mean_squared_error'])
        model_fg.compile(optimizer=tfk.optimizers.Adam(lr=0.0002),
        loss=['mean_squared_error', 'mean_squared_error', 'mean_squared_error', 'mean_squared_error'])
        model_h.summary()
        
        return model_h, model_fg
    

class TrainModel(Model):
    def __init__(self, img_dim, img_channel, *args, **kwargs):
        super(TrainModel, self).__init__(*args, **kwargs)
        self.muzero_h = MuZero_Model([3, 3, 3])
        self.muzero_g = MuZero_Model([3, 3, 3])
        self.muzero_f = MuZero_Model([2, 3, 3])
        #input_shape = tf.TensorShape([None, img_dim, img_dim, img_channel])
        ##input_shape_1 = tf.TensorShape([None, img_dim, img_dim, 128 + 1])
        #input_shape_2 = tf.TensorShape([None, img_dim, img_dim, 128])
        #self.muzero_h.build(input_shape)
        #self.muzero_g.build(input_shape_1)
        #self.muzero_f.build(input_shape_2)
    
    def call(self, inputs):
        action = inputs[:,:,:, -5:]
        inputs = inputs[:,:,:,:-5]
        state = self.muzero_h(inputs)
        policies, values, rewards = [], [], []
        for i in range(5):
            self.muzero_f(state)
            values.append(self.muzero_f.densed_1)
            policies.append(self.muzero_f.densed)
            #state = self.muzero_g(tf.concat([state, tf.reshape(np.array([[[action[i]]]], dtype=np.float32), shape=[1, len(action[i][0]), len(action[i][1]), 1])], axis=-1))
            state = self.muzero_g(tf.concat([state, action[:,:,:,i:i + 1]], axis=-1))
            rewards.append(self.muzero_g.densed)
        return [policies, values, rewards]

class Muzero(Model):
    def __init__(self, img_dim: int, img_channel: int, *args, **kwargs):
        super(Muzero, self).__init__(*args, **kwargs)
        self.inputs = Input(shape=(img_dim, img_dim, img_channel + 5))
        self.inputs_action = Input(shape=(img_dim, img_dim, 5))
        #self.inputs_action = [[[0 for i in range(9)] for u in range(9)] for s in range(5)]
        self.train_muzero = TrainModel(img_dim, img_channel)
    
    def get_layer_weight(self):
        return self.train_muzero.muzero_h.get_weights(), \
            self.train_muzero.muzero_f.get_weights(), self.train_muzero.muzero_g.get_weights()

    def compile_model(self):
        output = self.train_muzero(self.inputs)
        self.model = Model(inputs=[self.inputs], outputs=[output])
        self.model.compile(optimizer=tfk.optimizers.Adam(lr=0.0002),
        loss='mean_squared_error')
        #self.model.layers[1].summary()
        #model_h = MuZero_MCTS(9, 2)
        #model_h.h.set_weights(self.model.layers[1].layers[0].get_weights())
        #model_h.h.summary()
        #self.train_muzero.muzero_h.summary()
        #model_h.h.set_weights(self.train_muzero.muzero_h.get_weights())
        #sys.exit()
        return self.model

