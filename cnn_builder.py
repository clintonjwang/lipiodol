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
from keras.utils import np_utils

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
from os.path import *
import pandas as pd
import random
from scipy.misc import imsave
from skimage.transform import rescale
from niftiutils.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
import time

def build_unet(optimizer='adam', depth=4, base_f=8, nb_segs=2, dropout=.1, lr=.001):
	"""Main class for setting up a CNN. Returns the compiled model."""
	import importlib
	importlib.reload(cnnc)
	importlib.reload(config)

	C = config.Config()
	levels = []

	img = Input(shape=(C.dims[0], C.dims[1], C.dims[2]))
	current_layer = Reshape((C.dims[0], C.dims[1], C.dims[2], 1))(img)
	#current_layer = Reshape((1, C.dims[0], C.dims[1], C.dims[2]))(img)
	
	for layer_depth in range(depth):
		layer1 = cnnc.conv_block(current_layer, base_f*2**layer_depth, strides=1)
		dropl = layers.Dropout(dropout)(layer1)
		layer2 = cnnc.conv_block(dropl, base_f*2**(layer_depth+1))

		if layer_depth < depth - 1:
			current_layer = layers.MaxPooling3D((2,2,2))(layer2)
			levels.append([layer1, layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1, layer2])
			
	for layer_depth in range(depth-2, -1, -1):
		up_convolution = cnnc.up_conv_block(pool_size=(2,2,2), deconvolution=False,
											n_filters=current_layer._keras_shape[1])(current_layer)
		concat = layers.Concatenate(axis=-1)([up_convolution, levels[layer_depth][1]])
		current_layer = cnnc.conv_block(concat, levels[layer_depth][1]._keras_shape[-1]//2)
		current_layer = layers.Dropout(dropout)(current_layer)
		current_layer = cnnc.conv_block(current_layer, levels[layer_depth][1]._keras_shape[-1]//2)

	x = layers.Conv3D(nb_segs, (1,1,1), activation='softmax')(current_layer)
	#x = layers.Softmax(-1)(x)
	#x = layers.Permute((4,1,2,3))(x)
	model = Model(img, x)

	model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
	return model

def build_unet_multich(optimizer='adam', depth=3, base_f=32, dropout=.1, lr=.001):
	"""Main class for setting up a CNN. Returns the compiled model."""
	import importlib
	importlib.reload(cnnc)
	importlib.reload(config)

	C = config.Config()
	levels = []

	img = Input(shape=(*C.small_dims, 4))
	current_layer = img
	#current_layer = Reshape((C.dims[0], C.dims[1], C.dims[2], 1))(img)
	#current_layer = Reshape((1, C.dims[0], C.dims[1], C.dims[2]))(img)
	
	for layer_depth in range(depth):
		layer1 = cnnc.conv_block(current_layer, base_f*2**layer_depth, strides=1)
		dropl = layers.Dropout(dropout)(layer1)
		layer2 = cnnc.conv_block(dropl, base_f*2**(layer_depth+1))

		if layer_depth < depth - 1:
			current_layer = layers.MaxPooling3D((2,2,2))(layer2)
			levels.append([layer1, layer2, current_layer])
		else:
			current_layer = layer2
			levels.append([layer1, layer2])
			
	for layer_depth in range(depth-2, -1, -1):
		up_convolution = cnnc.up_conv_block(pool_size=(2,2,2), deconvolution=False,
											n_filters=current_layer._keras_shape[1])(current_layer)
		concat = layers.Concatenate(axis=-1)([up_convolution, levels[layer_depth][1]])
		current_layer = cnnc.conv_block(concat, levels[layer_depth][1]._keras_shape[-1]//2)
		current_layer = layers.Dropout(dropout)(current_layer)
		current_layer = cnnc.conv_block(current_layer, levels[layer_depth][1]._keras_shape[-1]//2)

	x = layers.Conv3D(2, (1,1,1), activation='softmax')(current_layer)
	#x = layers.Softmax(-1)(x)
	#x = layers.Permute((4,1,2,3))(x)
	model = Model(img, x)

	model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
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
	Z_test = test_ids = np.array(orig_data_dict[-1][order[train_samples:]])
	
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

def train_gen_lip(n=5):
	C = config.Config()

	lesion_ids = [z[:z.find("_lip")] for z in os.listdir(C.train_data_dir) if z.endswith("_lipdensity.npy")]
	while True:
		X_train = np.empty((n,*C.small_dims,4))
		Y_train = np.empty((n,*C.small_dims,2))

		for ix in range(n):
			lesion_id = random.choice(lesion_ids)
			x = np.stack([np.load(join(C.train_data_dir, lesion_id+"_%s.npy" % x)) for x in ["mrbl_art", "mrbl_sub", "mrbl_equ", "mrbl_tumor_mask"]], -1)
			y = np.load(join(C.train_data_dir, lesion_id+"_lipdensity.npy"))

			angle = random.uniform(-180,180)*math.pi/180
			x = tr.rotate(x, angle)
			y = tr.rotate(y.astype(float), angle)

			crops = list(map(int,[random.uniform(0,.1) * x.shape[0], random.uniform(.9,1) * x.shape[0]] + \
							[random.uniform(0,.1) * x.shape[1], random.uniform(.9,1) * x.shape[1]] + \
							[random.uniform(0,.1) * x.shape[2], random.uniform(.9,1) * x.shape[2]]))
			x = x[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5], :]
			y = y[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]

			x,_ = tr.rescale_img(x, C.small_dims)
			y,_ = tr.rescale_img(y, C.small_dims)
			y[y>0] = 1
			y = np_utils.to_categorical(y, 2)
			#y = np.expand_dims(y,-1)

			X_train[ix] = np.expand_dims(x,0)
			Y_train[ix] = np.expand_dims(y,0)

		yield X_train, Y_train

def train_gen_mri(n=1):
	C = config.Config()

	lesion_ids = [z[:z.find("_mri")] for z in os.listdir(C.train_data_dir) if z.endswith("_mri_art.npy")]
	while True:
		lesion_id = random.choice(lesion_ids)
		x = np.load(join(C.train_data_dir, lesion_id+"_mri_art.npy"))
		y = np.load(join(C.train_data_dir, lesion_id+"_mr_bl_liver_mask.npy"))

		angle = random.uniform(-20,20)*math.pi/180
		x = tr.rotate(x, angle)
		y = tr.rotate(y.astype(float), angle)

		crops = list(map(int,[random.uniform(0,.1) * x.shape[0], random.uniform(.9,1) * x.shape[0]] + \
						[random.uniform(0,.1) * x.shape[1], random.uniform(.9,1) * x.shape[1]] + \
						[random.uniform(0,.1) * x.shape[2], random.uniform(.9,1) * x.shape[2]]))
		x = x[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]
		y = y[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]

		x,_ = tr.rescale_img(x, C.dims)
		y,_ = tr.rescale_img(y, C.dims)
		y = np_utils.to_categorical(y, 2)
		yield np.expand_dims(x,0), np.expand_dims(y,0)

def train_gen_ct(n=1):
	C = config.Config()

	lesion_ids = [z[:z.find("_")] for z in os.listdir(C.train_data_dir) if z.endswith("_ct.npy")]
	while True:
		lesion_id = random.choice(lesion_ids)
		x = np.load(join(C.train_data_dir, lesion_id+"_ct.npy"))
		y = np.load(join(C.train_data_dir, lesion_id+"_mask.npy"))

		angle = random.uniform(-10,10)*math.pi/180
		x = tr.rotate(x, angle)
		y = tr.rotate(y.astype(float), angle)

		crops = list(map(int,[random.uniform(0,.1) * x.shape[0], random.uniform(.9,1) * x.shape[0]] + \
						[random.uniform(0,.1) * x.shape[1], random.uniform(.9,1) * x.shape[1]] + \
						[random.uniform(0,.1) * x.shape[2], random.uniform(.9,1) * x.shape[2]]))
		x = x[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]
		y = y[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]

		x,_ = tr.rescale_img(x, C.dims)
		y,_ = tr.rescale_img(y, C.dims)
		y = np_utils.to_categorical(y, 2)
		yield np.expand_dims(x,0), np.expand_dims(y,0)

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