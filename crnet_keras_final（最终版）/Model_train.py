import numpy as np
import random
import h5py
import sklearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from sklearn import model_selection
#from Model_define_tf import Encoder, Decoder, NMSE
from Model_define_tf_crnet_mod_from_914_to_916 import Encoder, Decoder, NMSE


# parameters
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2

    


def NMSE_LOSS(x, x_hat):
# =============================================================================
    constant = tf.fill(tf.shape(x),0.5)
    #constant = tf.fill(K.int_shape(x_hat),0.5)
    #b = tf.fill(K.int_shape(a),0.5)
    #x_real = x[:, :, :, 0]
    #x_imag = x[:, :, :, 1]
    #x_hat_real = x_hat[:, :, :, 0]
    #x_hat_imag = x_hat[:, :, :, 1]
# =============================================================================
    x_C = x - constant
    x_hat_C = x_hat - constant
    power = K.sum(abs(x_C) ** 2, axis=1)
    mse = K.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = K.mean(mse / power)
    return nmse


# =============================================================================
# 
# def NMSE_LOSS_2(x, x_hat):
#     x = x.numpy()
#     x_hat = x_hat.numpy()
#     x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
#     x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
#     x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
#     x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
#     x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
#     x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
#     power = np.sum(abs(x_C) ** 2, axis=1)
#     mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
#     nmse = np.mean(mse / power)
#     return nmse
# =============================================================================



# Model construction
# encoder model
Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
Encoder_output = Encoder(Encoder_input, feedback_bits)
encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')

# decoder model
Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
Decoder_output = Decoder(Decoder_input, feedback_bits)
decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")

# autoencoder model
autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
adam = keras.optimizers.Adam(lr=2e-3)
autoencoder.compile(optimizer=adam, loss='mse')
print(autoencoder.summary())




# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address+'/Hdata.mat', 'r') 
data = np.transpose(mat['H_train'])      # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
data = np.transpose(data, (0, 2, 3, 1))   # change to data_form: 'channel_last'
x_train, x_test = sklearn.model_selection.train_test_split(data, test_size=0.01, random_state=42)


learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.75)
#并且作为callbacks进入generator,开始训练

autoencoder.fit(x=x_train, y=x_train, batch_size=512, epochs=500, validation_data = (x_test,x_test),callbacks=[learning_rate_reduction])
# =============================================================================
# for iii in range(0,10):
#     autoencoder.fit(x=data, y=data, batch_size=256, epochs=50, validation_split=0.01)
#     y_test = autoencoder.predict(x_test, batch_size=512)
#     print('The test NMSE is ' + np.str(NMSE(x_test, y_test)))
#     # # save encoder
#     modelsave1 = './Modelsave/encoder_'+str(iii)+'.h5'
#     encoder.save(modelsave1)
#     # # save decoder
#     modelsave2 = './Modelsave/decoder_'+str(iii)+'.h5'
#     decoder.save(modelsave2)
#     
# =============================================================================

# =============================================================================
# for i in range(0,20):
#     seed = int(random.uniform(1,15))
#     x_train, x_test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=seed)
# 
# # model training
#     autoencoder.fit(x=x_train, y=x_train, batch_size=512, epochs=100, validation_split=0.1)
# 
# =============================================================================

# =============================================================================
# # # model save
# # # save encoder
# modelsave1 = './Modelsave/encoder.h5'
# encoder.save(modelsave1)
# # # save decoder
# modelsave2 = './Modelsave/decoder.h5'
# decoder.save(modelsave2)
# =============================================================================


# model test
# =============================================================================
# y_train= autoencoder.predict(x_train, batch_size=512)
# print('The train NMSE is ' + np.str(NMSE(x_train, y_train)))
# =============================================================================
y_test = autoencoder.predict(x_test, batch_size=512)
print('The test NMSE is ' + np.str(NMSE(x_test, y_test)))






