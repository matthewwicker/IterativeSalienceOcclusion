# Author: Matthew Wicker
# Contact: matthew.wicker@cs.ox.ac.uk
# Companion Code for CVPR2019 Submission: Analysis of 3D Deep Learning in an Adversarial Setting


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution3D, MaxPooling3D
from keras.layers import Conv3D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
import random
import numpy as np
from keras import backend as K

def VoxNet(classes=10, load_weights=True):
	K.clear_session()
	model = Sequential()
	model.add(Conv3D(input_shape=(32, 32, 32, 1), filters=32, kernel_size=(5,5,5), strides=(2, 2, 2)))
	model.add(Activation(LeakyReLU(alpha=0.1)))
	model.add(Dropout(rate=0.3))
	model.add(Conv3D(filters=32, kernel_size=(3,3,3)))
	model.add(Activation(LeakyReLU(alpha=0.1)))
	model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=None))
	model.add(Dropout(rate=0.4))
	model.add(Flatten())
	model.add(Dense(units=128, activation='relu'))
	model.add(Dropout(rate=0.5))
	model.add(Dense(units=classes, kernel_initializer='normal', activation='relu'))
	model.add(Activation("softmax"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	if(classes==40 and load_weights):
	    model.load_weights("Models/VoxNet-ModelNet40.h5")
	if(classes==10 and load_weights):
	    model.load_weights("Models/VoxNet-ModelNet10.h5")
	if(classes==2 and load_weights):
	    model.load_weights("Models/VoxNet-KITTI.h5")
	if(classes==3 and load_weights):
	    print("Loaded KITTI3")
	    model.load_weights("Models/VoxNet-KITTI3.h5")
	print model.summary()
	return model


from keras.models import Model

def predict(x_in, model):
    x_in = np.squeeze(x_in)
    x_in = np.swapaxes(np.asarray([[x_in]]), 1,-1)
    val = model.predict(x_in)
    val = np.squeeze(val)
    cl = np.argmax(val)
    return val[cl], cl


def get_max_pool(model, point_cloud):
	point_cloud = np.squeeze(point_cloud)
   	point_cloud = np.swapaxes(np.asarray([[point_cloud]]), 1,-1)
	layer_name = 'max_pooling3d_1'
	intermediate_layer_model = Model(inputs=model.input,
		                     outputs=model.get_layer(layer_name).output)
	inp = np.asarray(point_cloud)
	activations = intermediate_layer_model.predict(inp)
	return activations


def get_latent_activations(model, point_cloud):
	layer_name = 'activation_2'
	intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
	value_test = np.asarray([X_train[test_subject]])
	intermediate_output = intermediate_layer_model.predict(value_test)
	return intermediate_output


def get_critical_set(model, point_cloud):
	latent = get_latent_activations(model, point_cloud)[0]
	critical_set = np.argmax(latent, axis=0)
	critical_set = set(critical_set)
	return critical_set


def get_critical_set_bb(model, point_cloud):
	return []
