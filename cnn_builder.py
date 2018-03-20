"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/lipiodol`
Author: David G Ellis (https://github.com/ellisdg/3DUnetCNN)
"""

import keras.backend as K
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Concatenate, Flatten, Dropout, Lambda
from keras.layers import SimpleRNN, Conv2D, MaxPooling2D, ZeroPadding3D, Activation, ELU, TimeDistributed, Permute, Reshape
from keras.layers.normalization import BatchNormalization
import keras.layers as layers
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import argparse
import copy
import config
import niftiutils.cnn_components as cnnc
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import math
from math import log, ceil
import numpy as np
import operator
import os
import pandas as pd
import random
from scipy.misc import imsave
from skimage.transform import rescale
from niftiutils.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
import time

def build_ct_unet(optimizer='adam', depth=3, base_f=16, nb_segs=2, lr=.001):
	"""Main class for setting up a CNN. Returns the compiled model."""

	C = config.Config()
	levels = []

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2]))
	current_layer = Reshape((1, C.dims[0], C.dims[1], C.dims[2]))(img)
	
	for layer_depth in range(depth):
		layer1 = cnnc.conv_block(current_layer, base_f*2**layer_depth)
		layer2 = cnnc.conv_block(layer1, base_f*2**(layer_depth+1))

		if layer_depth < depth - 1:
			current_layer = layers.MaxPooling3D((2,2,2))(layer2)
			levels.append([layer1, layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1, layer2])
			
	for layer_depth in range(depth-2, -1, -1):
		up_convolution = cnnc.up_conv_block(pool_size=(2,2,2), deconvolution=False,
											n_filters=current_layer._keras_shape[1])(current_layer)
		concat = layers.Concatenate(axis=1)([up_convolution, levels[layer_depth][1]])
		current_layer = cnnc.conv_block(concat, levels[layer_depth][1]._keras_shape[1])
		current_layer = cnnc.conv_block(current_layer, levels[layer_depth][1]._keras_shape[1])

	output = layers.Conv3D(nb_segs, (1,1,1))(current_layer)
	model = Model(img, output)

	model.compile(optimizer=Adam(lr=lr), loss=dice_coefficient_loss, metrics=['accuracy'])
	return model

def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)

def get_cnn_data(n=4):
	"""Subroutine to run CNN
	n is number of real samples, n_art is number of artificial samples
	Z_test is filenames"""

	C = config.Config()

	nb_classes = len(C.classes_to_include)
	orig_data_dict, num_samples = _collect_unaug_data()

	if C.train_frac is None:
		train_samples = num_samples - C.test_num
	else:
		train_samples = round(num_samples*C.train_frac)
	
	order = np.random.permutation(list(range(num_samples)))

	X_test = np.array(orig_data_dict[0][order[train_samples:]])
	Y_test = np.array(orig_data_dict[1][order[train_samples:]])
	Z_test = np.array(orig_data_dict[-1][order[train_samples:]])
	
	X_train_orig = np.array(orig_data_dict[0][order[:train_samples]])
	Y_train_orig = np.array(orig_data_dict[1][order[:train_samples]])
	Z_train_orig = np.array(orig_data_dict[-1][order[:train_samples]])

	if run_2d:
		train_generator = _train_generator_func_2d(test_ids, n=n, n_art=n_art)
	else:
		train_generator = _train_generator_func(test_ids, n=n)

	return X_test, Y_test, train_generator, num_samples, [X_train_orig, Y_train_orig], [Z_test, Z_train_orig]

####################################
### Training Submodules
####################################

def _train_generator_func(test_ids, n=1):
	"""n is the number of samples from each class, n_art is the number of artificial samples"""

	C = config.Config()

	voi_df = drm.get_voi_dfs()[0]

	#avg_X2 = {}
	#for cls in orig_data_dict:
	#	avg_X2[cls] = np.mean(orig_data_dict[cls][1], axis=0)
	patient_info_df = pd.read_csv(C.patient_info_path)
	patient_info_df["AccNum"] = patient_info_df["AccNum"].astype(str)

	num_classes = len(C.classes_to_include)
	x = np.empty((n, C.dims[0], C.dims[1], C.dims[2]))
	y = np.zeros((n, 2, C.dims[0], C.dims[1], C.dims[2]))

	train_cnt = 0

	img_fns = os.listdir(C.aug_dir+cls)
	while n > 0:
		img_fn = random.choice(img_fns)
		lesion_id = img_fn[:img_fn.rfind('_')]
		if lesion_id not in test_ids[cls]:
			x[train_cnt] = np.load(C.aug_dir+cls+"\\"+img_fn)
			if C.hard_scale:
				x1[train_cnt] = vm.scale_intensity(x1[train_cnt], 1, max_int=2, keep_min=False)
			
			y[train_cnt] = get_mask(x[train_cnt], ??)
			
			train_cnt += 1
			if train_cnt % (n+n_art) == 0:
				break

		yield np.array(x), np.array(y)

def _separate_phases(X, non_imaging_inputs=False):
	"""Assumes X[0] contains imaging and X[1] contains dimension data.
	Reformats such that X[0:2] has 3 phases and X[3] contains dimension data.
	Image data still is 5D (nb_samples, 3D, 1 channel).
	Handles both 2D and 3D images"""
	
	if non_imaging_inputs:
		dim_data = copy.deepcopy(X[1])
		img_data = X[0]
		
		if len(X[0].shape)==5:
			X[1] = np.expand_dims(X[0][:,:,:,:,1], axis=4)
			X += [np.expand_dims(X[0][:,:,:,:,2], axis=4)]
			X += [dim_data]
			X[0] = np.expand_dims(X[0][:,:,:,:,0], axis=4)
		
		else:
			X[1] = np.expand_dims(X[0][:,:,:,1], axis=3)
			X += [np.expand_dims(X[0][:,:,:,2], axis=3)]
			X += [dim_data]
			X[0] = np.expand_dims(X[0][:,:,:,0], axis=3)
	
	else:
		X = np.array(X)
		if len(X.shape)==5:
			X = [np.expand_dims(X[:,:,:,:,0], axis=4), np.expand_dims(X[:,:,:,:,1], axis=4), np.expand_dims(X[:,:,:,:,2], axis=4)]
		else:
			X = [np.expand_dims(X[:,:,:,0], axis=3), np.expand_dims(X[:,:,:,1], axis=3), np.expand_dims(X[:,:,:,2], axis=3)]

	return X

def _collect_unaug_data():
	"""Return dictionary pointing to X (img data) and Z (filenames) and dictionary storing number of samples of each class."""

	C = config.Config()
	orig_data_dict = {}
	num_samples = {}
	voi_df = drm.get_voi_dfs()[0]
	#voi_df = voi_df[voi_df["run_num"] <= C.test_run_num]
	patient_info_df = pd.read_csv(C.patient_info_path)
	patient_info_df["AccNum"] = patient_info_df["AccNum"].astype(str)

	for cls in C.classes_to_include:
		x = np.empty((10000, C.dims[0], C.dims[1], C.dims[2], C.nb_channels))
		z = []

		if C.dual_img_inputs:
			x2 = np.empty((10000, *C.context_dims, C.nb_channels))
		elif C.non_imaging_inputs:
			x2 = np.empty((10000, C.num_non_image_inputs))

		for index, lesion_id in enumerate(voi_df[voi_df["cls"] == cls].index):
			img_path = os.path.join(C.orig_dir, cls, lesion_id+".npy")
			try:
				x[index] = np.load(img_path)
				if C.hard_scale:
					x[index] = vm.scale_intensity(x[index], 1, max_int=2)#, keep_min=True)
			except:
				raise ValueError(img_path + " not found")
			z.append(lesion_id)
			
			if C.dual_img_inputs:
				tmp = np.load(os.path.join(C.crops_dir, cls, lesion_id+".npy"))
				x2[index] = tr.rescale_img(tmp, C.context_dims)[0]

			elif C.non_imaging_inputs:
				voi_row = voi_df.loc[lesion_id]
				patient_row = patient_info_df[patient_info_df["AccNum"] == voi_row["acc_num"]]
				x2[index] = get_non_img_inputs(voi_row, patient_row)

		x.resize((index+1, C.dims[0], C.dims[1], C.dims[2], C.nb_channels)) #shrink first dimension to fit
		if C.dual_img_inputs or C.non_imaging_inputs:
			x2.resize((index+1, *x2.shape[1:]))
			orig_data_dict[cls] = [x, x2, np.array(z)]
		else:
			orig_data_dict[cls] = [x, np.array(z)]

		num_samples[cls] = index + 1
		
	return orig_data_dict, num_samples

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert DICOMs to npy files and transfer voi coordinates from excel to csv.')
	parser.add_argument('-m', '--max_runs', type=int, help='max number of runs to allow')
	#parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite')
	args = parser.parse_args()

	run_fixed_hyperparams(max_runs=args.max_runs)