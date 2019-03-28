
# coding: utf-8

# In[ ]:


import numpy as np
import h5py

train_path = '../Data/KITTI_train/kitti_train.h5'
test_path = '../Data/KITTI_train/kitti_test.h5'
def load_data(path):
    f = h5py.File(path, 'r')
    X_train = np.asarray(list(f['data']))
    Y_train = np.asarray(list(f['label']))

    # Normalize the data

    for i in X_train:
        i -= np.mean(i, axis=0)

    m = max(X_train.min(), X_train.max(), key=abs)
    X_train /= m # This puts the data in [0,1]
    return X_train, Y_train

X_train, Y_train = load_data(train_path)
X_test, Y_test = load_data(train_path)


# In[ ]:


print X_train.shape
print Y_train.shape


# In[ ]:


import matplotlib.pyplot as plt

#y_test_hist = [np.argmax(i) for i in Y_test]
#y_train_hist = [np.argmax(i) for i in Y_train]

#plt.hist(y_train_hist)
#plt.show()

#plt.hist(y_test_hist)
#plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

index = 666

print("Label: %s"%(Y_train[5]))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')

X,Y,Z = np.hsplit(X_train[index],3)
scat = ax.scatter(X,Y,Z)

max_range = np.asarray([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()-X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()-Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()-Z.min())

for xb,yb,zb in zip(Xb,Yb,Zb):
    ax.plot([xb],[yb],[zb], 'r')

plt.grid()
plt.show()


# In[ ]:



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

def mat_mul(A, B):
    return tf.matmul(A, B)

# number of points in each sample
num_points = 1024

# number of categories
k = 2

# define optimizer
adam = optimizers.Adam(lr=0.001, decay=0.7)

# ------------------------------------ Pointnet Architecture
# input_Transformation_net
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
c = Dense(256, activation='relu')(global_feature)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(128, activation='relu')(c)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(k, activation='softmax')(c)
prediction = Flatten()(c)
# --------------------------------------------------end of pointnet

# print the model summary
model = Model(inputs=input_points, outputs=prediction)
print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


history = model.fit(x=X_train, y=Y_train, batch_size=32, 
                    epochs=25, verbose=1, validation_data=(X_train, Y_train))


# In[ ]:


from keras.models import model_from_json
import os
#model_json = model.to_json()
#with open("voxnet_kitti.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("pointnet_kitti.h5")
print("Saved model to disk")

