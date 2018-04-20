import config
import copy
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
		
	for ix,row in df.sort_values([column], ascending=False).iterrows():
		save_dir = join(root_dir, "%d_%s" % (row[column]*100, ix))
		
		patient_id = ix
		paths = lm.get_paths(patient_id, target_dir, check_valid=False)

		mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
		mribl_art_path, mribl_pre_path, mribl_sub_path, \
		mribl_tumor_mask_path, mribl_liver_mask_path, \
		mribl_enh_mask_path, mribl_nec_mask_path, \
		mri30d_art_path, mri30d_pre_path, \
		mri30d_tumor_mask_path, mri30d_liver_mask_path, \
		mri30d_enh_mask_path, mri30d_nec_mask_path, \
		ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
		ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
		midlip_mask_path, ball_midlip_mask_path, \
		highlip_mask_path, ball_highlip_mask_path = paths
		
		if mask_type is not None:
			masks.create_dcm_with_mask(eval(img_type), eval(mask_type), save_dir,
									   overwrite=True, padding=1.5, window=window)
		else:
			img = hf.nii_load(eval(img_type))
			if window=="ct":
				img = tr.apply_window(img)
			hf.create_dicom(img, save_dir, overwrite=overwrite)


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

def draw_sub_and_depo(lesion_id, target_dir, save_dir, include_FU=False):
	paths = lm.get_paths(lesion_id, target_dir)

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, mribl_sub_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths

	ART = masks.crop_img_to_mask_vicinity(mribl_art_path, mribl_tumor_mask_path,.1)
	PRE = masks.crop_img_to_mask_vicinity(mribl_pre_path, mribl_tumor_mask_path,.1)
	CT = masks.crop_img_to_mask_vicinity(ct24_path, ct24_tumor_mask_path,.1)
	CT = tr.apply_window(CT)

	if include_FU:
		art=masks.crop_img_to_mask_vicinity(mri30d_art_path, mri30d_tumor_mask_path,.1)
		pre=masks.crop_img_to_mask_vicinity(mri30d_pre_path, mri30d_tumor_mask_path,.1)
		hf.draw_multi_slices([ART-PRE, CT, art-pre], save_path=join(save_dir, lesion_id), width=3, dpi=400)
	else:
		hf.draw_multi_slices([ART-PRE, CT], save_path=join(save_dir, lesion_id), width=4)


###########################
### Draw figures
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
	return [check_column(lesion_id, master_df, "0=well delineated, 1=infiltrative", {0: "Focal", 1: "Infiltrative"}),
			check_column(lesion_id, master_df, "HCC(0), ICC(1), other(2)", {0: "HCCs", 1: "ICCs", 2: "Metastases"}),
			check_column(lesion_id, master_df, "selective=0", {0: "Selective TACE", 1: "Lobar TACE"}),
			check_homogeneous(lesion_id, master_df, modality),
			check_sparse(lesion_id, master_df, modality),
			check_rim(lesion_id, master_df, modality)]

def check_homogeneous(lesion_id, df, modality):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "enhancing_vol%",
			legend_names=["Homogeneous\nenhancement", "Heterogeneous\nenhancement"],
			criteria_pos=lambda x: x > .75, restriction="Focal")

	elif modality == "ct24":
		return check_feature(lesion_id, df, "lipcoverage_vol%",
			legend_names=["Homogeneous\ndeposition", "Heterogeneous\ndeposition"],
			criteria_pos=lambda x: x > .85, restriction="Focal")

def check_sparse(lesion_id, df, modality, restriction=None):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "enhancing_vol%",
			legend_names=["Sparse enhancement", "Non-sparse, heterogeneous\nenhancement"],
			criteria_pos=lambda x: x < .25, criteria_neg=lambda x: (x>=.25) & (x<=.85),
			restriction=restriction)

	elif modality == "ct24":
		return check_feature(lesion_id, df[df["lipcoverage_vol%"] <= .85], "lipcoverage_vol%",
			legend_names=["Sparse deposition", "Non-sparse, heterogeneous\ndeposition"],
			criteria_pos=lambda x: x < .2,#, criteria_neg=lambda x: (x>=.25) & (x<=.85),
			restriction=restriction)

def check_rim(lesion_id, df, modality):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "rim_enhancing%",
			legend_names=["Rim enhancement", "Non-rim, heterogeneous\nenhancement"],
			criteria_pos=lambda x: x > .5, restriction="Focal")

	elif modality == "ct24":
		return check_feature(lesion_id, df[df["lipcoverage_vol%"] <= .85], "rim_lipiodol%",
			legend_names=["Rim deposition", "Non-rim, heterogeneous\ndeposition"],
			criteria_pos=lambda x: x > .28, restriction="Focal")

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