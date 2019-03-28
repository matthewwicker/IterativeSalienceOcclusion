
# coding: utf-8

# In[2]:


import numpy as np
import h5py

path = '../Data/KITTI_train/kitti_train.h5'

f = h5py.File(path, 'r')
X_train = f['data']
Y_train = f['label']


# In[3]:


print X_train.shape
print Y_train.shape


# In[8]:


from tqdm import trange
# now we need to voxelize that point cloud...
def voxelize(dim, data):
    # uncomment below if you have not already normalized your object to [0,1]^3
    m = max(data.min(), data.max(), key=abs)
    data /= m # This puts the data in [0,1]
    data *= (dim/2)-1 # This puts the data in [0,dim]
    data += (dim/2) 
    data = np.asarray([[int(i[0]), int(i[1]), int(i[2])] for i in data])
    data = np.unique(data, axis=1)
    retval = np.zeros((dim, dim, dim))
    for i in data:
        retval[i[0]][i[1]][i[2]] = 1
    retval = np.asarray([retval])
    return retval

X_train = [voxelize(32, i) for i in X_train]
X_train = np.asarray(X_train)
X_train = np.reshape(X_train, (-1, 32, 32, 32, 1))
print X_train.shape


# In[10]:


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

num_classes = 2

# Defining VoxNet in Keras 2
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
model.add(Dense(units=num_classes, kernel_initializer='normal', activation='relu'))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
model.summary()


# In[13]:


history = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=25, verbose=1, shuffle="batch")


# In[ ]:


from keras.models import model_from_json
import os
model_json = model.to_json()
with open("voxnet_kitti.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("voxnet_kitti.h5")
print("Saved model to disk")

