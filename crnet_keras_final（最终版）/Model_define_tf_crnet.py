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

from tensorflow.keras.layers import Activation, Reshape, Lambda, dot, add
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras import backend as K

#from tensorflow.keras.utils.generic_utils import get_custom_objects

def swish(inputs):
    return (K.sigmoid(inputs) * inputs)
    
def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6
    
#函数绑定，将激活函数swish和h_swish添加到keras框的Activation类中
#get_custom_objects().update({'swish': Activation(swish)})    
#get_custom_objects().update({'h_swish': Activation(h_swish)})


def non_local_block(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).

    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.

    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x



#This part realizes the quantization and dequantization operations.
#The output of the encoder must be the bitstream.


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

    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)  #cast change the type of data

    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)  #func can change the numpy to tensor inp:the input of func

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
    
def convbn(input_x,output,conv): #convbn(x,2,3)
    x = layers.Conv2D(output, conv, padding = 'SAME')(input_x)
    x = layers.BatchNormalization()(x)
    x = swish(x)
    return x

def ecn(input_x,output,conv): #convbn(x,2,3)
    #layers.UpSampling2D(size=(2,2))(input_x)
    x = layers.AveragePooling2D(pool_size=(2,2))(input_x) #make maxpooling and concatenate
    #x = layers.pooling.MaxPooling2D(pool_size=(2,2))(input_x) 
    x = layers.Conv2D(output, conv, padding = 'SAME')(x)
    x = layers.BatchNormalization()(x)
    x = swish(x)
    return x

def dn(input_x,output,conv): #convbn(x,2,3)
    #layers.UpSampling2D(size=(2,2))(input_x)
    x = layers.UpSampling2D(size=(2,2))(input_x)
    x = layers.Conv2D(output, conv, padding = 'SAME')(x)
    x = layers.BatchNormalization()(x)
    x = swish(x)
    return x


def crblock(input_x): #convbn(x,2,3)
    x1 = convbn(input_x,32,7)
    x1 = convbn(x1,32,(1,9))
    x1 = convbn(x1,32,(9,1))
    #print('x1',x1.shape)
    x2 = convbn(input_x,32,(1,7))
    x2 = convbn(x2,32,(7,1))
    #print('x2',x2.shape)
    x = layers.concatenate([x1,x2],axis=3)
    #print('x',x.shape)
    x = swish(x)
    x = layers.Conv2D(32,3,padding = 'SAME')(x)
    #print('xout',x.shape)
    x = layers.Add()([x,input_x])
    x = swish(x)
    
    return x

def encoder1(input_x): #convbn(x,2,3)
    x1 = convbn(input_x,32,3)
    x1 = convbn(x1,64,(1,9))
    x1 = convbn(x1,32,(9,1))
    
    return x1

def encoder2(input_x): #convbn(x,2,3)
    x1 = convbn(input_x,32,7)
    x1 = convbn(x1,64,(1,9))
    x1 = convbn(x1,32,(1,9))
    
    return x1

def decoder(input_x): #convbn(x,2,3)
    x1 = convbn(input_x,32,5)
    x1 = crblock(x1)
    x1 = crblock(x1)
    
    return x1


def add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = swish(y)
    return y

def dense_residual_block(y):
        #layers_concat = list()
        #layers_concat.append(y)

    y1 = add_common_layers(y)
    y1 = Conv2D(8, (3, 3), padding='same')(y1)
        #layers_concat.append(y)
    y2 = layers.concatenate([y,y1], axis=3)
    y2 = add_common_layers(y2)
    y2 = Conv2D(16, (3, 3), padding='same')(y2)
        #layers_concat.append(y)

    y3 = layers.concatenate([y,y1,y2], axis=3)
    y3 = add_common_layers(y3)
    y3 = Conv2D(2, (3, 3), padding='same')(y3)
        #layers_concat.append(y)

    y4 = layers.concatenate([y,y1,y2,y3], axis=3)
        

        #print(y.shape)
    return y4


def RefineNet(input_x):  # convbn(x,2,3)
    x1 = convbn(input_x, 32, 3)
    x1 = convbn(x1, 16, 5)
    x1 = convbn(x1, 32, 5)
    x1 = convbn(x1, 64, 3)

    x = layers.Add()([x1, input_x])

    return x

def Encoder(x,feedback_bits):
    B=4
    #def ecn(input_x,output,conv): #convbn(x,2,3)
    with tf.compat.v1.variable_scope('Encoder'):
        #print(x.shape)
        x1 = encoder1(x)
        x2 = encoder2(x)
        x = layers.concatenate([x1,x2],axis=3)
        #x = non_local_block(x, compression=2, mode='embedded' )
        #x = convbn(x,2,3)
        x = ecn(x,128,7)
        x = ecn(x,256,3)
        x = ecn(x,512,2)
        x = convbn(x, 256, 2)
        x = layers.Flatten()(x)
        x = layers.Dense(units=int(feedback_bits/B), activation='sigmoid')(x)
        encoder_output = QuantizationLayer(B)(x)
    return encoder_output


def Decoder(x,feedback_bits):
    B=4
    decoder_input = DeuantizationLayer(B)(x)
    x = tf.keras.layers.Reshape((-1, int(feedback_bits/B)))(decoder_input)
    x = layers.Dense(256*2*4, activation='sigmoid')(x) # try linear 

    x = layers.Reshape((2, 4, 256))(x)
    x = convbn(x, 512, 2)
    x = dn(x, 256, 3)
    x = dn(x, 128, 7)
    x = dn(x, 64, 9)
    #x = convbn(x,32,3)
    #x = non_local_block(x, compression=2, mode='embedded' )
    #x = RefineNet(x)
    #x = RefineNet(x)
    x = decoder(x)
    x = dense_residual_block(x)
    #x = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")
    #x_ini = layers.Reshape((16, 32, 2))(x)

# =============================================================================
#     for i in range(3):
#         x = layers.Conv2D(8, 3, padding = 'SAME',activation="relu")(x_ini)
#         x = layers.Conv2D(16,3, padding = 'SAME',activation="relu")(x)
#         x = layers.Conv2D(2, 3, padding = 'SAME',activation="relu")(x)
#         x_ini = keras.layers.Add()([x_ini, x])
# =============================================================================


    decoder_output = layers.Conv2D(2, 3, padding = 'SAME',activation="sigmoid")(x)

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
    return {"QuantizationLayer":QuantizationLayer,"DeuantizationLayer":DeuantizationLayer,"swish":swish}