# Author: Matthew Wicker
# Companion Code for paper: Analysis of 3D Deep Learning in an Adversarial Setting
# CVPR 2019


"""
This file impliments the PointNet model (Qi et. al. 2017) in keras
It is aware of weights that are saved in the Models directory of this
repository. So if you would like to modify/retrain this model, then
please ensure the weights and architecture are changed accordingly.
"""

import h5py
import numpy as np
import numpy as np
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
from keras import backend as K
import copy
def mat_mul(A, B):
    return tf.matmul(A, B)

"""
This function declares the PointNet architecture.
@ Param classes - Integer, defining the number of classes that the
		 model will be predicting.
@ Param load_weights - Boolean, if classes is set for 10 or 40 we 
		will load pretrained weights into the model.
"""
def PointNet(classes=40, load_weights=True, num_points =2048):
	num_points = num_points
	input_points = Input(shape=(num_points, 3))
	x = Convolution1D(64, 1, activation='relu',
		          input_shape=(num_points, 3))(input_points)
	x = BatchNormalization()(x)
	x = Convolution1D(128, 1, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Convolution1D(1024, 1, activation='relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size=num_points)(x)
	x = Dense(512, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dense(256, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
	input_T = Reshape((3, 3))(x)

	# forward net
	g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
	g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
	g = BatchNormalization()(g)
	g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
	g = BatchNormalization()(g)

	# feature transform net
	f = Convolution1D(64, 1, activation='relu')(g)
	f = BatchNormalization()(f)
	f = Convolution1D(128, 1, activation='relu')(f)
	f = BatchNormalization()(f)
	f = Convolution1D(1024, 1, activation='relu')(f)
	f = BatchNormalization()(f)
	f = MaxPooling1D(pool_size=num_points)(f)
	f = Dense(512, activation='relu')(f)
	f = BatchNormalization()(f)
	f = Dense(256, activation='relu')(f)
	f = BatchNormalization()(f)
	f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
	feature_T = Reshape((64, 64))(f)

	# forward net
	g = Lambda(mat_mul, arguments={'B': feature_T})(g)
	g = Convolution1D(64, 1, activation='relu')(g)
	g = BatchNormalization()(g)
	g = Convolution1D(128, 1, activation='relu')(g)
	g = BatchNormalization()(g)
	g = Convolution1D(1024, 1, activation='relu')(g)
	g = BatchNormalization()(g)

	# global_feature
	global_feature = MaxPooling1D(pool_size=num_points)(g)

	# point_net_cls
	c = Dense(512, activation='relu')(global_feature)
	c = BatchNormalization()(c)
	c = Dropout(rate=0.7)(c)
	c = Dense(256, activation='relu')(c)
	c = BatchNormalization()(c)
	c = Dropout(rate=0.7)(c)
	c = Dense(classes, activation='softmax')(c)
	prediction = Flatten()(c)
	
	model = Model(inputs=input_points, outputs=prediction)
	

	model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

	if(classes == 40 and load_weights):
    		model.load_weights('Models/PointNet-ModelNet40.h5')
	if(classes == 10 and load_weights):
    		model.load_weights('Models/PointNet-ModelNet10.h5')
	if(classes == 2 and load_weights):
    		model.load_weights('Models/PointNet-KITTI.h5')
	if(classes == 3 and load_weights):
    		model.load_weights('Models/PointNet-KITTI3.h5')
	print model.summary()

	return model


def predict(x_in, model):
    x_in = np.squeeze(x_in)
    val = model.predict(np.asarray([x_in]))
    val = np.squeeze(val)
    cl = np.argmax(val)
    return val[cl], cl

"""
This method returns the activations of the max pooling layer for
the specified inputs. Expects the following input
@Param - model, the Keras model that is outputted from the PointNet() function
@Param - point_cloud, the point cloud input that we want the maxpooling layer for
"""
def get_max_pool(model, point_cloud):
	layer_name = 'max_pooling1d_3'
	intermediate_layer_model = Model(inputs=model.input,
		                     outputs=model.get_layer(layer_name).output)
	inp = np.asarray(point_cloud)
	activations = intermediate_layer_model.predict(inp)
	return activations


def get_latent_activations(model, point_cloud):
	layer_name = 'batch_normalization_15'
	intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
	value_test = np.asarray([point_cloud])
	intermediate_output = intermediate_layer_model.predict(value_test)
	return intermediate_output


def get_critical_set(model, point_cloud):
	latent = get_latent_activations(model, point_cloud)[0]
	critical_set = np.argmax(latent, axis=0)
	critical_set = set(critical_set)
	return critical_set


def get_critical_set_bb(model, point_cloud):
    critical_set = []
    values = []
    #v_init, c_init = predict(point_cloud, model)
    pc = copy.deepcopy(point_cloud)	
    for i in range(len(pc)):
        val = copy.deepcopy(pc[i])
        pc[i] = [0.0,0.0,0.0]
        v, c = predict(pc, model)
        values.append(v)
        pc[i] = val
    unique = np.unique(values,return_index=True)
    return unique[1]
		








