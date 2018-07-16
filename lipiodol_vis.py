import config
import copy
import cv2
import importlib
import lipiodol_methods as lm
import niftiutils.masks as masks
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.registration as reg
import niftiutils.visualization as vis
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
	importlib.reload(masks)
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

###########################
### Draw tumor pngs
###########################

def draw_unreg_fig(img_path, mask_path, save_path, color, modality, midslice=True):
	img,D = hf.nii_load(img_path)
	mask = masks.get_mask(mask_path, D, img.shape)
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
	mask = masks.get_mask(mask_path)
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
		mask = masks.get_mask(P[mod]['enh'], D, I.shape)
		mask = hf.crop_nonzero(mask, C)[0]
	else:
		mask = np.zeros(art.shape)

	tumor_mask = masks.get_mask(P[mod]['tumor'], D, I.shape)
	tumor_mask = hf.crop_nonzero(tumor_mask, C)[0]

	sub_w_mask = vis.create_contour_img(sub, [tumor_mask, mask])

	vis.display_sequence([pre[...,sl], art[...,sl], equ[...,sl], sub[...,sl],
		sub_w_mask, mask[...,sl]], 2, 3,
		join(save_dir, "%s_%s.png" % (lesion_id, mod)))

def draw_reg_seq(lesion_id, target_dir, save_dir):
	importlib.reload(hf)
	importlib.reload(masks)
	P = lm.get_paths_dict(lesion_id, target_dir)

	out_img = []
	bl_img = masks.crop_img_to_mask_vicinity(P['mrbl']['sub'], P['mrbl']['tumor'], .5, add_mask_cont=True)
	fu_img = masks.crop_img_to_mask_vicinity(P['mr30']['sub'], P['mr30']['tumor'], .5, add_mask_cont=True)
	ct_img = masks.crop_img_to_mask_vicinity(P['ct24']['img'], P['ct24']['tumor'], .5, add_mask_cont=True, window=[0,300])
	bl_Tx,D = hf.nii_load(P['ct24Tx']['mrbl']['art'])
	fu_Tx = hf.nii_load(P['ct24Tx']['mr30']['art'])[0]

	tumor_mask = masks.get_mask(P['ct24Tx']['crop']['tumor'])
	sl = bl_Tx.shape[-1]//2

	if exists(P['ct24Tx']['mrbl']['enh'] + ".off"):
		bl_M = masks.get_mask(P['ct24Tx']['mrbl']['enh'], D, bl_Tx.shape)
	else:
		bl_M = np.zeros(bl_Tx.shape)

	if exists(P['ct24Tx']['mr30']['enh'] + ".off"):
		fu_M = masks.get_mask(P['ct24Tx']['mr30']['enh'], D, bl_Tx.shape)
	else:
		fu_M = np.zeros(bl_Tx.shape)

	mask_overlay = np.stack([bl_M[...,sl], np.zeros(bl_M.shape[:2]), fu_M[...,sl]], -1)

	bl_Tx_cont = vis.create_contour_img(bl_Tx[...,sl], [tumor_mask[...,sl], bl_M[...,sl]], colors=[(0,255,0), (255,0,0)])
	fu_Tx_cont = vis.create_contour_img(fu_Tx[...,sl], [tumor_mask[...,sl], fu_M[...,sl]], colors=[(0,255,0), (0,0,255)])

	#if exists(P['ct24Tx']['crop']['midlip'] + ".off"):
	#	ct_M = masks.get_mask(P['ct24Tx']['crop']['midlip'], D, bl_Tx.shape)
	#else:
	#	ct_M = np.zeros(bl_Tx.shape)
	#mask_overlay2 = np.stack([bl_M[...,sl]*fu_M[...,sl]/fu_M.max(), ct_M[...,sl], bl_M[...,sl]*(1-fu_M[...,sl]/fu_M.max())], -1)

	vis.display_sequence([bl_img, fu_img, ct_img,#bl_img[...,bl_img.shape[-1]//2], fu_img[...,fu_img.shape[-1]//2], #ct_img[...,ct_img.shape[-1]//2]
		bl_Tx_cont, #np.transpose(ct_Tx_cont, (1,0,2)), 
		fu_Tx_cont, mask_overlay],#, mask_overlay2],
		2, 3, join(save_dir, "%s.png" % lesion_id))

###########################
### Draw figure
###########################

def check_feature(lesion_id, df, column, legend_names, criteria_pos, criteria_neg=None, restriction=None):
	if lesion_id not in df.index:
		return np.nan
	if criteria_neg is None:
		criteria_neg = lambda x: ~criteria_pos(x) & ~np.isnan(x)

	if criteria_pos(df.loc[lesion_id, column]):
		if restriction=="WD":
			cnt = (criteria_pos(df[column]) & (df["0=well delineated, 1=infiltrative"]==0)).sum()
		elif restriction=="Infiltrative":
			cnt = (criteria_pos(df[column]) & (df["0=well delineated, 1=infiltrative"]==1)).sum()
		else:
			cnt = criteria_pos(df[column]).sum()
		return legend_names[0] + " (n=%d)" % cnt
	
	elif criteria_neg(df.loc[lesion_id, column]):
		if restriction=="WD":
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
		return [check_column(lesion_id, master_df, "0=well delineated, 1=infiltrative", {0: "Well-delineated", 1: "Infiltrative"}),
				check_column(lesion_id, master_df, "HCC(0), ICC(1), other(2)", {0: "HCCs", 1: "ICCs", 2: "Metastases"}),
				check_column(lesion_id, master_df, "selective=0", {0: "Selective TACE", 1: "Lobar TACE"})]
	elif modality == "ct24":
		return [check_column(lesion_id, master_df, "0=well delineated, 1=infiltrative", {0: "Well-delineated", 1: "Infiltrative"}),
				check_column(lesion_id, master_df, "HCC(0), ICC(1), other(2)", {0: "HCCs", 1: "ICCs", 2: "Metastases"}),
				check_column(lesion_id, master_df, "selective=0", {0: "Selective TACE", 1: "Lobar TACE"}),
				check_homogeneous(lesion_id, master_df, modality),
				check_sparse(lesion_id, master_df, modality),
				check_rim(lesion_id, master_df, modality)]

def check_homogeneous(lesion_id, df, modality):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "enhancing_vol",
			legend_names=["Homogeneous\nenhancement", "Heterogeneous\nenhancement"],
			criteria_pos=lambda x: x > .75, restriction="Well-delineated")

	elif modality == "ct24":
		return check_feature(lesion_id, df, "lipcoverage_vol",
			legend_names=["Homogeneous\ndeposition", "Heterogeneous\ndeposition"],
			criteria_pos=lambda x: x >= .8, restriction="Well-delineated")

def check_sparse(lesion_id, df, modality, restriction=None):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "enhancing_vol",
			legend_names=["Sparse enhancement", "Non-sparse, heterogeneous\nenhancement"],
			criteria_pos=lambda x: x < .25, criteria_neg=lambda x: (x>=.25) & (x<.8),
			restriction=restriction)

	elif modality == "ct24":
		return check_feature(lesion_id, df[df["lipcoverage_vol"] < .8], "lipcoverage_vol",
			legend_names=["Sparse deposition", "Non-sparse, heterogeneous\ndeposition"],
			criteria_pos=lambda x: x < .2, criteria_neg=lambda x: (x>=.2) & (x<.8),
			restriction=restriction)

def check_rim(lesion_id, df, modality):
	if modality == "mrbl":
		return check_feature(lesion_id, df, "rim_enhancing",
			legend_names=["Rim enhancement", "Non-rim, heterogeneous\nenhancement"],
			criteria_pos=lambda x: x > .5, restriction="Well-delineated")

	elif modality == "ct24":
		return check_feature(lesion_id, df[df["lipcoverage_vol"] < .8], "rim_lipiodol",
			legend_names=["Rim deposition", "Non-rim, heterogeneous\ndeposition"],
			criteria_pos=lambda x: x > .5, restriction="Well-delineated")

def check_column(lesion_id, df, column, mapping, restriction=None):
	if np.isnan(df.loc[lesion_id, column]):
		return np.nan
	else:
		if restriction=="WD":
			cnt = ((df[column]==df.loc[lesion_id, column]) & \
				   (df["0=well delineated, 1=infiltrative"]==0)).sum()
		elif restriction=="Infiltrative":
			cnt = ((df[column]==df.loc[lesion_id, column]) & \
				   (df["0=well delineated, 1=infiltrative"]==1)).sum()
		else:
			cnt = (df[column]==df.loc[lesion_id, column]).sum()
		return mapping[df.loc[lesion_id, column]] + " (n=%d)" % cnt