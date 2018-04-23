import config
import copy
import cv2
import importlib
import lipiodol_methods as lm
import niftiutils.masks as masks
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.registration as reg
import numpy as np
import random
import math
from math import pi, radians, degrees
import matplotlib.pyplot as plt
import glob
import shutil
import os
from os.path import *
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation
from skimage.morphology import ball, label

###########################
### Create tumor DICOMs
###########################

def write_ranked_imgs(df, target_dir, column, img_type, root_dir, overwrite=False, mask_type=None, window=None):
	if not exists(root_dir):
		os.makedirs(root_dir)
		
	for ix,row in df.dropna(subset=[column]).sort_values([column], ascending=False).iterrows():
		save_dir = join(root_dir, "%d_%s" % (row[column]*100, ix))
		
		patient_id = ix
		P = lm.get_paths_dict(patient_id, target_dir)
		
		if mask_type is not None:
			masks.create_dcm_with_mask(eval(img_type), eval(mask_type), save_dir,
									   overwrite=True, padding=1.5, window=window)
		else:
			img = hf.nii_load(eval(img_type))
			if window=="ct":
				img = tr.apply_window(img)
			hf.create_dicom(img, save_dir, overwrite=overwrite)

def check_qEASL_threshold(lesion_id, target_dir):
	P = lm.get_paths_dict(lesion_id, target_dir)
	P['mrbl']

###########################
### Draw tumor pngs
###########################

def draw_unreg_fig(img_path, mask_path, save_path, color, modality, midslice=True):
	img,D = hf.nii_load(img_path)
	mask,_ = masks.get_mask(mask_path, D, img.shape)
	nz = np.argwhere(mask)

	pad = [img.shape[0]//5, img.shape[1]//5]
	sl1 = slice(max(nz[:,0].min()-pad[0],0), nz[:,0].max()+pad[0])
	sl2 = slice(max(nz[:,1].min()-pad[1],0), nz[:,1].max()+pad[1])
	img = np.transpose(img[sl1,sl2], (1,0,2))
	mask = np.transpose(mask[sl1,sl2], (1,0,2))
	sl1, sl2 = nz[:,-1].min(), nz[:,-1].max()

	if midslice:
		RNG = [(sl1+sl2)//2]
	else:
		RNG = range(sl1,sl2, max((sl2-sl1)//10,1))
		
	if not exists(dirname(save_path)):
		os.makedirs(dirname(save_path))

	for sl in RNG:
		plt.close()
		if modality=="mr":
			plt.imshow(img[...,sl], cmap='gray')
		elif modality=="ct":
			plt.imshow(img[...,sl], cmap='gray', vmin=30, vmax=250)
		plt.contour(mask[:,:,sl], colors=color, alpha=.4)
		plt.axis('off')
		if midslice:
			plt.savefig(save_path+".png", dpi=100, bbox_inches='tight')	
		else:
			plt.savefig(save_path+"_%d.png" % sl, dpi=100, bbox_inches='tight')	

def draw_reg_fig(img_path, mask_path, save_path, color, modality):
	img,_ = hf.nii_load(img)
	mask,_ = masks.get_mask(mask_path)
	img = np.transpose(img, (1,0,2))
	mask = np.transpose(mask, (1,0,2))
	
	for sl in range(img.shape[-1]//5+1,img.shape[-1]*4//5, max(img.shape[-1]//8,1) ):
		plt.close()
		if modality=="mr":
			plt.imshow(img[...,sl], cmap='gray')
		elif modality=="ct":
			plt.imshow(img[...,sl], cmap='gray', vmin=30, vmax=250)
		plt.contour(mask[:,:,sl], colors=color, alpha=.4)
		plt.axis('off')
		plt.savefig(save_path+"_%d.png" % sl, dpi=100, bbox_inches='tight')

def draw_sub_and_depo(lesion_id, target_dir, save_dir, include_FU=False, padding=.3):
	importlib.reload(lm)
	P = lm.get_paths_dict(lesion_id, target_dir)

	mod='mrbl'
	ART = masks.crop_img_to_mask_vicinity(P[mod]['art'], P[mod]['tumor'], padding)
	PRE = masks.crop_img_to_mask_vicinity(P[mod]['pre'], P[mod]['tumor'], padding)
	CT = masks.crop_img_to_mask_vicinity(P['ct24']['img'], P['ct24']['tumor'], padding)
	CT = tr.apply_window(CT)

	if include_FU:
		mod='mr30'
		art = masks.crop_img_to_mask_vicinity(P[mod]['art'], P[mod]['tumor'], padding)
		pre = masks.crop_img_to_mask_vicinity(P[mod]['pre'], P[mod]['tumor'], padding)
		hf.draw_multi_slices([ART-PRE, CT, art-pre], save_path=join(save_dir, lesion_id), width=3, dpi=400)
	else:
		hf.draw_multi_slices([ART-PRE, CT], save_path=join(save_dir, lesion_id), width=4)

def draw_mrseq_with_mask(lesion_id, target_dir, save_dir, mod='mrbl'):
	importlib.reload(masks)
	P = lm.get_paths_dict(lesion_id, target_dir)

	out_img = []
	art,C = masks.crop_img_to_mask_vicinity(P[mod]['art'], P[mod]['tumor'], .5, return_crops=True)
	pre = masks.crop_img_to_mask_vicinity(P[mod]['pre'], P[mod]['tumor'], .5)
	equ = masks.crop_img_to_mask_vicinity(P[mod]['equ'], P[mod]['tumor'], .5)
	sub = art - pre

	sl = art.shape[-1]//2

	I,D = hf.nii_load(P[mod]['art'])
	if exists(P[mod]['enh'] + ".off"):
		mask = masks.get_mask(P[mod]['enh'], D, I.shape)[0]
		mask = hf.crop_nonzero(mask, C)[0]
	else:
		mask = np.zeros(art.shape)

	tumor_mask = masks.get_mask(P[mod]['tumor'], D, I.shape)[0]
	tumor_mask = hf.crop_nonzero(tumor_mask, C)[0]

	sub_w_mask = create_contour_img(sub[...,sl], [tumor_mask[...,sl], mask[...,sl]])

	"""tumor_mask = (tumor_mask/tumor_mask.max()*255).astype('uint8')
				_,thresh = cv2.threshold(tumor_mask[:,:,sl],127,255,0)
				contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
				tumor_cont = cv2.drawContours(np.zeros((art.shape[0],art.shape[1],3),'uint8'), contours, -1, (0,255,0), 1)
			
				sub_w_mask = sub[...,sl] - sub[...,sl].min()
				sub_w_mask = (sub_w_mask/sub_w_mask.max()*255).astype('uint8')
				sub_w_mask = sub_w_mask * (tumor_cont[...,1] == 0)
			
				if mask[...,sl].sum() == 0:
					#sub_w_mask = sub[...,sl]
					sub_w_mask = np.stack([sub_w_mask, sub_w_mask, sub_w_mask], -1)
					sub_w_mask += tumor_cont
				else:
					mask = (mask/mask.max()*255).astype('uint8')
					_,thresh = cv2.threshold(mask[:,:,sl],127,255,0)
					contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
					cont = cv2.drawContours(np.zeros((art.shape[0],art.shape[1],3),'uint8'), contours, -1, (255,0,0), 1)
					sub_w_mask = sub_w_mask * (cont[...,0] == 0)
			
					tumor_cont[cont[...,0] != 0] = 0
					sub_w_mask = np.stack([sub_w_mask, sub_w_mask, sub_w_mask], -1)
					sub_w_mask += tumor_cont
					sub_w_mask += cont"""

	out_img.append(pre[...,sl])
	out_img.append(art[...,sl])
	out_img.append(equ[...,sl])
	out_img.append(sub[...,sl])
	out_img.append(np.transpose(sub_w_mask, (1,0,2)))
	out_img.append(mask[...,sl])
	#img = np.transpose(img, (1,0,2))
	#mask = np.transpose(mask, (1,0,2))

	for ix in range(4):
		plt.subplot(231+ix)
		hf._plot_without_axes(out_img[ix])
	plt.subplot(231+4)
	fig = plt.imshow(out_img[4])
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.subplot(231+5)
	hf._plot_without_axes(out_img[5])
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig(join(save_dir, "%s_%s.png" % (lesion_id, mod)), dpi=150, bbox_inches='tight')

def display_sequence(rows, cols, save_path):
	out_img.append(pre[...,sl])
	out_img.append(art[...,sl])
	out_img.append(equ[...,sl])
	out_img.append(sub[...,sl])
	out_img.append(np.transpose(sub_w_mask, (1,0,2)))
	out_img.append(mask[...,sl])
	#img = np.transpose(img, (1,0,2))
	#mask = np.transpose(mask, (1,0,2))

	for ix in range(4):
		plt.subplot(231+ix)
		hf._plot_without_axes(out_img[ix])
	plt.subplot(231+4)
	fig = plt.imshow(out_img[4])
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	plt.subplot(231+5)
	hf._plot_without_axes(out_img[5])
	plt.subplots_adjust(wspace=0, hspace=0)
	plt.savefig(join(save_dir, "%s_%s.png" % (lesion_id, mod)), dpi=150, bbox_inches='tight')

def create_contour_img(img_sl, mask_sl, colors=[(0,255,0), (255,0,0)]):
	if type(mask_sl) != list:
		mask_sl = [mask_sl]

	if mask_sl[0].max() == 0:
		return img_sl

	mask = (mask_sl[0]/mask_sl[0].max()*255).astype('uint8')
	_,thresh = cv2.threshold(mask[:,:,sl],127,255,0)
	contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
	cont1 = cv2.drawContours(np.zeros((img.shape[0],img.shape[1],3),'uint8'), contours, -1, colors[0], 1)

	img = img_sl - img_sl.min()
	img = (img/img.max()*255).astype('uint8')
	img = img * (cont1[...,1] == 0)

	if len(mask_sl) == 1 or mask_sl[1].max() == 0:
		img = np.stack([img, img, img], -1)
		img += cont1
	else:
		mask = (mask_sl[1]/mask_sl[1].max()*255).astype('uint8')
		_,thresh = cv2.threshold(mask[:,:,sl],127,255,0)
		contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
		cont2 = cv2.drawContours(np.zeros((art.shape[0],art.shape[1],3),'uint8'), contours, -1, colors[1], 1)

		img = img * (cont2[...,0] == 0)
		cont2[cont1[...,0] != 0] = 0
		img = np.stack([img, img, img], -1)
		img += cont1 + cont2

	return img

###########################
### Draw figure
###########################

def check_feature(lesion_id, df, column, legend_names, criteria_pos, criteria_neg=None, restriction=None):
	if lesion_id not in df.index:
		return np.nan
	if criteria_neg is None:
		criteria_neg = lambda x: ~criteria_pos(x) & ~np.isnan(x)

	if criteria_pos(df.loc[lesion_id, column]):
		if restriction=="Focal":
			cnt = (criteria_pos(df[column]) & (df["0=well delineated, 1=infiltrative"]==0)).sum()
		elif restriction=="Infiltrative":
			cnt = (criteria_pos(df[column]) & (df["0=well delineated, 1=infiltrative"]==1)).sum()
		else:
			cnt = criteria_pos(df[column]).sum()
		return legend_names[0] + " (n=%d)" % cnt
	
	elif criteria_neg(df.loc[lesion_id, column]):
		if restriction=="Focal":
			cnt = (criteria_neg(df[column]) & (df["0=well delineated, 1=infiltrative"]==0)).sum()
		elif restriction=="Infiltrative":
			cnt = (criteria_neg(df[column]) & (df["0=well delineated, 1=infiltrative"]==1)).sum()
		else:
			cnt = criteria_neg(df[column]).sum()
		return legend_names[1] + " (n=%d)" % cnt
	else:
		return np.nan

def get_df_entry(lesion_id, master_df, modality):
	if modality == "mrbl":
		return [check_column(lesion_id, master_df, "0=well delineated, 1=infiltrative", {0: "Focal", 1: "Infiltrative"}),
				check_column(lesion_id, master_df, "HCC(0), ICC(1), other(2)", {0: "HCCs", 1: "ICCs", 2: "Metastases"}),
				check_column(lesion_id, master_df, "selective=0", {0: "Selective TACE", 1: "Lobar TACE"})]
	elif modality == "ct24":
		return [check_column(lesion_id, master_df, "0=well delineated, 1=infiltrative", {0: "Focal", 1: "Infiltrative"}),
				check_column(lesion_id, master_df, "HCC(0), ICC(1), other(2)", {0: "HCCs", 1: "ICCs", 2: "Metastases"}),
				check_column(lesion_id, master_df, "selective=0", {0: "Selective TACE", 1: "Lobar TACE"}),
				check_homogeneous(lesion_id, master_df, modality),
				check_sparse(lesion_id, master_df, modality),
				check_rim(lesion_id, master_df, modality)]

def check_homogeneous(lesion_id, df, modality):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "enhancing_vol",
			legend_names=["Homogeneous\nenhancement", "Heterogeneous\nenhancement"],
			criteria_pos=lambda x: x > .75, restriction="Focal")

	elif modality == "ct24":
		return check_feature(lesion_id, df, "lipcoverage_vol",
			legend_names=["Homogeneous\ndeposition", "Heterogeneous\ndeposition"],
			criteria_pos=lambda x: x >= .8, restriction="Focal")

def check_sparse(lesion_id, df, modality, restriction=None):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "enhancing_vol",
			legend_names=["Sparse enhancement", "Non-sparse, heterogeneous\nenhancement"],
			criteria_pos=lambda x: x < .25, criteria_neg=lambda x: (x>=.25) & (x<.8),
			restriction=restriction)

	elif modality == "ct24":
		return check_feature(lesion_id, df[df["lipcoverage_vol"] < .8], "lipcoverage_vol",
			legend_names=["Sparse deposition", "Non-sparse, heterogeneous\ndeposition"],
			criteria_pos=lambda x: x < .2, criteria_neg=lambda x: (x>=.25) & (x<.8),
			restriction=restriction)

def check_rim(lesion_id, df, modality):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "rim_enhancing",
			legend_names=["Rim enhancement", "Non-rim, heterogeneous\nenhancement"],
			criteria_pos=lambda x: x > .5, restriction="Focal")

	elif modality == "ct24":
		return check_feature(lesion_id, df[df["lipcoverage_vol"] < .8], "rim_lipiodol",
			legend_names=["Rim deposition", "Non-rim, heterogeneous\ndeposition"],
			criteria_pos=lambda x: x > 25, restriction="Focal")

def check_column(lesion_id, df, column, mapping, restriction=None):
	if np.isnan(df.loc[lesion_id, column]):
		return np.nan
	else:
		if restriction=="Focal":
			cnt = ((df[column]==df.loc[lesion_id, column]) & \
				   (df["0=well delineated, 1=infiltrative"]==0)).sum()
		elif restriction=="Infiltrative":
			cnt = ((df[column]==df.loc[lesion_id, column]) & \
				   (df["0=well delineated, 1=infiltrative"]==1)).sum()
		else:
			cnt = (df[column]==df.loc[lesion_id, column]).sum()
		return mapping[df.loc[lesion_id, column]] + " (n=%d)" % cnt