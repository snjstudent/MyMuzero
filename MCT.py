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
import GameBoard_tmp
import numpy as np
import random
import sys
import math
C_1, C_2 = 1.25, 19652
K = 3
discount_per = 0.8


global model
board = GameBoard_tmp.GameBoard()


class node:
    def __init__(self, state, action, policy, reward: float = 0.0):
        self.action = action
        self.state = state
        self.visit_counts = 0
        self.mean_value = 0.0
        self.policy = policy
        self.reward = reward
        self.child_nodes = []
    
    def selection(self):
        def pucb_values():
            sum_visit = sum([child.visit_counts for child in self.child_nodes])
            return self.mean_value + (self.policy * sum_visit / (1 + self.visit_counts) * (C_1 + math.log((sum_visit + C_2 + 1) / C_2)))
        return self.child_nodes[np.argmax(pucb_values())]
        

    def expansion(self):
        def expansion_child_node(nodes, child_node_tmp, action, num: int = 0):
            legal_action_list = board.legal_actions
            pred_datas = [model.predict([nodes.state, tf.concat([nodes.state, np.array([[action]], \
            dtype=np.float32).reshape([1, 9, 9, 1])], axis=-1)]) for action in legal_action_list]
            child_nodes = [node(pred_data[0], legal_action_list[i], pred_data[3], pred_data[1]) for i, pred_data in enumerate(pred_datas)]
            nodes.child_nodes.extend(child_nodes)
            child_node_tmp[0] = child_nodes if num == 0 else child_node_tmp[0] + child_nodes
    
        child_node_tmp = [[]]
        expansion_child_node(self, child_node_tmp, self.action)
        for _ in range(K - 2): 
            for u, child_node in enumerate(child_node_tmp[0]):
                expansion_child_node(child_node, child_node_tmp, child_node.action, u)
    
    def backup_parts_1(self, value, rewards):
        if self.child_nodes:
            self.selection().backup_parts_1(value, rewards)
        else:
            value = self.mean_value
        rewards.append(self.reward)
    
    def backup_parts_2(self, G_k, max_value: float, min_value: float):
        if self.child_nodes:
            self.selection().backup_parts_2(G_k, max_value, min_value)
        self.mean_value += (((self.visit_counts * self.mean_value) + G_k[-1]) / (self.visit_counts + 1)) - self.mean_value
        G_k = G_k[:-1]
        max_value += self.mean_value - max_value if self.mean_value > max_value else max_value
        min_value += self.mean_value - min_value if min_value > self.mean_value else min_value
        self.visit_counts += 1

    def normalize_mean_value(self, max_value, min_value):
        if self.child_nodes:
            self.selection().normalize_mean_value(max_value, min_value)
        self.mean_value = ((self.mean_value - min_value) / (max_value - min_value)) - self.mean_value
        
    def backup(self):
        value, max_value, min_value = 0.0, 0.0, 10.0 ** 10
        rewards, G_k = [], []
        self.backup_parts_1(value, rewards)
        for i in range(K - 1):
            discounts = [discount_per ** (t) for t in range(K - i)]
            G_k.append(sum(np.dot(np.array([discounts]), np.array(rewards[i:])) + (discount_per ** (K - 1 - i) * value)))
        self.backup_parts_2(G_k, max_value, min_value)
        self.normalize_mean_value(max_value, min_value)