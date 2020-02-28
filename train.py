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
import Plan_Act

class Train:
    def __init__(self):
        self.plan_act = Plan_Act.Plan_Act()
        self.muzero = model_muzero.Muzero(256, 3)
    
    def train(self):
        self.plan_act.act()
        
        
        
