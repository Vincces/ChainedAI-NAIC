#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:41:59 2020

@author: user
"""

"""
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU,LSTM


#This part realizes the quantization and dequantization operations.
#The output of the encoder must be the bitstream.

# image params
img_height = 16
img_width = 32
img_channels = 2
img_total = img_height*img_width*img_channels #each i and q are 32*32
# network params
residual_num = 2  #define how many RefinedNet, now is two
encoded_dim = 128  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32


def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, 4:]).reshape(-1,
                                                                                                            Num_.shape[
                                                                                                                1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)


@tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)

    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config

@tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, B,**kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config



# =============================================================================
# def Encoder(x,feedback_bits):
#     def add_common_layers(y):
#         y = BatchNormalization()(y) #BatchNormalization ÊÇ±ê×¼»¯²ã£¬½â¾ö´«Í³µÄÉñ¾­ÍøÂçÑµÁ·ÐèÒªÎÒÃÇÈËÎªµÄÈ¥Ñ¡Ôñ²ÎÊý£¬±ÈÈçÑ§Ï°ÂÊ¡¢²ÎÊý³õÊ¼»¯¡¢È¨ÖØË¥¼õÏµÊý¡¢Drop out±ÈÀý µÄÎÊÌâ£¬²¢ÄÜÌá¸ßËã·¨µÄÊÕÁ²ËÙ¶È¡£
#         y = LeakyReLU()(y) #¸ß¼¶activation function
#         return y
#     B=4
#     with tf.compat.v1.variable_scope('Encoder'):
#         x = layers.Conv2D(2, (3, 3), padding='same')(x)
#         
#         #x = layers.Flatten()(x)
#         
#         x = add_common_layers(x)
#         
#         x = Reshape((img_total,))(x)
#         
# # =============================================================================
# #         x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
# #         x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
# # =============================================================================
#         
#         x = layers.Dense(units=int(feedback_bits/B), activation='sigmoid')(x)
#         encoder_output = QuantizationLayer(B)(x)
#     return encoder_output
# def Decoder(x,feedback_bits):
#     def add_common_layers(y):
#         y = BatchNormalization()(y) #BatchNormalization ÊÇ±ê×¼»¯²ã£¬½â¾ö´«Í³µÄÉñ¾­ÍøÂçÑµÁ·ÐèÒªÎÒÃÇÈËÎªµÄÈ¥Ñ¡Ôñ²ÎÊý£¬±ÈÈçÑ§Ï°ÂÊ¡¢²ÎÊý³õÊ¼»¯¡¢È¨ÖØË¥¼õÏµÊý¡¢Drop out±ÈÀý µÄÎÊÌâ£¬²¢ÄÜÌá¸ßËã·¨µÄÊÕÁ²ËÙ¶È¡£
#         y = LeakyReLU()(y) #¸ß¼¶activation function
#         return y
#     def residual_block_decoded(y):
#         shortcut = y
#         y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
#         y = add_common_layers(y)
#         
#         y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_last')(y)
#         y = add_common_layers(y)
#         
#         y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_last')(y) #data_format='channels_last'
#         y = BatchNormalization()(y)
# 
#         y = add([shortcut, y])
#         y = LeakyReLU()(y)
# 
#         return y
#     B=4
#     decoder_input = DeuantizationLayer(B)(x)
#     x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
#     x = layers.Dense(1024, activation='sigmoid')(x)
#     x = layers.Reshape((16, 32, 2))(x)
#     for i in range(residual_num):
#         x = residual_block_decoded(x)
# 
# # =============================================================================
# #     for i in range(3):
# #         x = layers.Conv2D(8, 3, padding = 'SAME',activation="relu")(x_ini)
# #         x = layers.Conv2D(16,3, padding = 'SAME',activation="relu")(x)
# #         x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
# #         x_ini = keras.layers.Add()([x_ini, x])
# # =============================================================================
# 
# 
#     #decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")(x)
#     decoder_output = layers.Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_last")(x)
# 
#     return decoder_output
# =============================================================================

def Encoder(x,feedback_bits):
    B=4
    with tf.compat.v1.variable_scope('Encoder'):
        x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
        #x = BatchNormalization()(x)
        #x = LeakyReLU()(x)
        x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=int(feedback_bits/B), activation='sigmoid')(x)
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output
def Decoder(x,feedback_bits):
    B=4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
    x = layers.Dense(1024, activation='sigmoid')(x)
    x_ini = layers.Reshape((16, 32, 2))(x)

    for i in range(5):
        shortcut = x_ini
        x = layers.Conv2D(8, 3, padding = 'SAME',activation="relu")(x_ini)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = layers.Conv2D(16,3, padding = 'SAME',activation="relu")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = keras.layers.Add()([shortcut, x])
        x_ini = LeakyReLU()(x)


    decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")(x_ini)

    return decoder_output


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def Score(NMSE):
    score = 1-NMSE
    return score

# Return keywords of your own custom layers to ensure that model
# can be successfully loaded in test file.
def get_custom_objects():
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer}
