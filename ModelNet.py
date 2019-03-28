# Author: Matthew Wicker
# Contact: matthew.wicker@cs.ox.ac.uk

from keras.utils import to_categorical
import h5py
import numpy as np
from tqdm import tqdm

def voxelize(dim, data):
    # uncomment below if you have not already normalized your object to [0,1]^3
    #m = max(x.min(), x.max(), key=abs)
    #data /= m # This puts the data in [0,1]
    data *= (dim/2) # This puts the data in [0,dim]
    data += (dim/2) 
    data = np.asarray([[int(i[0]), int(i[1]), int(i[2])] for i in data])
    data = np.unique(data, axis=1)
    retval = np.zeros((dim, dim, dim))
    for i in data:
        retval[i[0]][i[1]][i[2]] = 1
    retval = np.asarray([retval])
    return retval


def _load_h5(paths):
	for i in range(len(paths)):
	    fh5 = h5py.File(paths[0], 'r')
	    data = fh5['data'][:]
	    label = fh5['label'][:]
	    fh5.close()
	    if(i != 0):
		d = np.append(d, data, axis=0)
		l = np.append(l, label, axis=0)
	    else:
		d = data
		l = label
	X = d
	Y = to_categorical(l)
	return X, Y

def loadmodelnet(classes=10, vox=False):
	if(classes==40):
		test_files = ['Data/ModelNet40_test/ply_data_test0.h5',
		 	      'Data/ModelNet40_test/ply_data_test1.h5']

		train_files = ['Data/ModelNet40_train/ply_data_train0.h5',
				'Data/ModelNet40_train/ply_data_train1.h5',
				'Data/ModelNet40_train/ply_data_train2.h5',
				'Data/ModelNet40_train/ply_data_train3.h5',
				'Data/ModelNet40_train/ply_data_train4.h5']
	elif(classes==10):
		train_files = ['Data/ModelNet10_train/modelnet10_train.h5']
		test_files = ['Data/ModelNet10_test/modelnet10_test.h5']

	X_train, Y_train = _load_h5(train_files)
	X_test, Y_test = _load_h5(test_files)
	
	if(vox==True):
		X_train = [voxelize(32, i) for i in tqdm(X_train, desc='Voxelizing Train')]
		X_test = [voxelize(32, i) for i in tqdm(X_test, desc='Voxelizing Test')]


	return X_train, Y_train, X_test, Y_test


