import config
import copy
import importlib
import lipiodol_methods as lm
import mahotas.features as mah
import niftiutils.masks as masks
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.registration as reg
import numpy as np
import random
import math
from math import pi, radians, degrees
import pandas as pd
import glob
import shutil
import os
import time
from os.path import *
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation
from skimage.morphology import ball, label

###########################
### Assess correlations
###########################

"""def enhancing_to_nec(lesion_id, target_dir, liplvls=[0,75,160,215]):
	P = lm.get_paths_dict(lesion_id, target_dir)

	mrbl_enh = masks.get_mask(P['ball']['mrbl']['enh'])[0]
	mrbl_enh = mrbl_enh/mrbl_enh.max()
	mr30d_enh = masks.get_mask(P['ball']['mr30']['enh'])[0]
	mr30d_enh = mr30d_enh/mr30d_enh.max()
	mrbl_nec = masks.difference(P['ball']['mask'], P['ball']['mrbl']['enh'])
	mrbl_nec = mrbl_nec/mrbl_nec.max()
	mr30d_nec = masks.difference(P['ball']['mask'], P['ball']['mr30']['enh'])
	mr30d_nec = mr30d_nec/mr30d_nec.max()
	
	ct24 = hf.nii_load(P['ball']['ct24']['img'])[0]
	
	resp = mrbl_enh * mr30d_nec
	unresp = mrbl_enh * mr30d_enh
	prog = mrbl_nec * mr30d_enh
	stable = mrbl_nec * mr30d_nec
	
	lips = [[np.sum([(ct24 * resp) > liplvls[i]]),
			  np.sum([(ct24 * unresp) > liplvls[i]]), 
			  np.sum([(ct24 * prog) > liplvls[i]]), 
			  np.sum([(ct24 * stable) > liplvls[i]]),
			  np.sum([ct24 > liplvls[i]])]
			 for i in range(len(liplvls))]
	
	lips = [[l[0]/(l[0]+l[1]), l[2]/(l[2]+l[3]), (l[0]+l[1])/l[4], (l[1]+l[2])/l[4]] for l in lips]
	return lips"""

def vascular_to_deposition(lesion_id, target_dir, liplvls, exclude_small=True):
	P = lm.get_paths_dict(lesion_id, target_dir)
	L = liplvls + [10000]

	M = masks.get_mask(P['ct24Tx']['crop']['tumor'])[0]
	M = M/M.max()
	if exists(P['ct24Tx']['mrbl']['enh'] + ".off"):
		mrbl_enh = masks.get_mask(P['ct24Tx']['mrbl']['enh'])[0]
		mrbl_enh = mrbl_enh/mrbl_enh.max()
	else:
		mrbl_enh = np.zeros(M.shape)
	mrbl_nec = M-mrbl_enh
	mrbl_nec[mrbl_nec < 0] = 0

	ct24 = hf.nii_load(P['ct24Tx']['crop']['img'])[0]
	ct24[ct24 <= 0] = 1
	enh_ct = ct24 * mrbl_enh
	nec_ct = ct24 * mrbl_nec
	lips_enh=[]
	lips_nec=[]
	lips_avg=[]
	for i in range(len(L)-1):
		den = np.sum(mrbl_enh)
		if den == 0 or (exclude_small and den <= 50): #den / V <= .05
			lips_enh.append(np.nan)
		else:
			lips_enh.append(np.nansum(enh_ct > L[i]) / den) # & (enh_ct <= L[i+1])

		den = np.sum(mrbl_nec)
		if den == 0 or (exclude_small and den <= 50): #den / V <= .05
			lips_nec.append(np.nan)
		else:
			lips_nec.append(np.nansum(nec_ct > L[i]) / den) # & (nec_ct <= L[i+1])

		lips_avg.append(np.sum(ct24[M != 0] > L[i]) / M.sum())

	return lips_nec + lips_enh + lips_avg

def lip_to_response(lesion_id, target_dir, liplvls, exclude_small=True):
	P = lm.get_paths_dict(lesion_id, target_dir)
	L = liplvls + [10000]

	M = masks.get_mask(P['ct24Tx']['crop']['tumor'])[0]
	M = M/M.max()
	if exists(P['ct24Tx']['mrbl']['enh'] + ".off"):
		mrbl_enh = masks.get_mask(P['ct24Tx']['mrbl']['enh'])[0]
		mrbl_enh = mrbl_enh/mrbl_enh.max()
	else:
		return [np.nan]*(len(liplvls)+1)

	if exists(P['ct24Tx']['mr30']['enh'] + ".off"):
		mr30d_enh = masks.get_mask(P['ct24Tx']['mr30']['enh'])[0]
		mr30d_nec = M * (1 - mr30d_enh/mr30d_enh.max())
	else:
		mr30d_enh = np.zeros(M.shape)
		mr30d_nec = M

	ct24 = hf.nii_load(P['ct24Tx']['crop']['img'])[0]
	ct24[ct24 < 0] = 1

	enh_ct = ct24 * mrbl_enh
	#V = np.sum(enh_ct != 0)
	#resp = mrbl_enh * mr30d_nec
	#resp_ct = ct24 * resp

	lips = []
	#z=np.zeros((3,3))
	#z[1,1]=1
	#B2 = np.stack([z,np.ones((3,3)),z],-1)
	#B2 = np.ones((3,3,1))
	cross = np.zeros((3,3))
	cross[1,:] = 1
	cross[:,1] = 1
	cross = np.expand_dims(cross, -1)
	for i in range(len(L)-1):
		lip_segment = (enh_ct > L[i]) & (enh_ct <= L[i+1])
		lip_segment = binary_closing(binary_opening(lip_segment, cross))

		den = lip_segment.sum()
		if den == 0 or (exclude_small and den <= 25):#(den / V <= .05 or den <= 50)):
			lips.append(np.nan)
		else:
			lips.append(np.sum(mr30d_nec*lip_segment / den))

	return lips + [(mrbl_enh*mr30d_nec).sum()/mrbl_enh.sum()]

def vascular_to_deposition_ball(lesion_id, target_dir, liplvls, exclude_small=True):
	P = lm.get_paths_dict(lesion_id, target_dir)
	L = liplvls + [10000]

	mrbl_enh = masks.get_mask(P['ball']['mrbl']['enh'])[0]
	mrbl_enh = mrbl_enh/mrbl_enh.max()
	ball = masks.get_mask(ball_mask_path)[0]
	ball = ball/ball.max()
	mrbl_nec = ball-mrbl_enh
	mrbl_nec[mrbl_nec < 0] = 0

	ct24 = hf.nii_load(P['ball']['ct24']['img'])[0]
	ct24[ct24 <= 0] = 1
	enh_ct = ct24 * mrbl_enh
	nec_ct = ct24 * mrbl_nec
	lips_enh=[]
	lips_nec=[]
	for i in range(len(L)-1):
		den = np.sum(mrbl_enh)
		if den == 0 or (exclude_small and den <= 50): #den / V <= .05
			lips_enh.append(np.nan)
		else:
			lips_enh.append(np.nansum([(enh_ct > L[i])]) / den) # & (enh_ct <= L[i+1])

		den = np.sum(mrbl_nec)
		if den == 0 or (exclude_small and den <= 50): #den / V <= .05
			lips_nec.append(np.nan)
		else:
			lips_nec.append(np.nansum([(nec_ct > L[i])]) / den) # & (nec_ct <= L[i+1])

	return lips_nec + lips_enh

def lip_to_response_ball(lesion_id, target_dir, liplvls, exclude_small=True):
	paths = lm.get_paths(lesion_id, target_dir, check_valid=False)
	L = liplvls + [10000]

	mrbl_enh = masks.get_mask(P['ball']['mrbl']['enh'])[0]
	mrbl_enh = mrbl_enh/mrbl_enh.max()

	mr30d_nec = masks.difference(ball_mask_path, P['ball']['mr30']['enh'])
	mr30d_nec = mr30d_nec/mr30d_nec.max()

	ct24 = hf.nii_load(P['ball']['ct24']['img'])[0]
	resp = mrbl_enh * mr30d_nec

	enh_ct = ct24 * mrbl_enh
	V = np.sum(enh_ct > 0)

	resp_ct = ct24 * resp
	lips=[]
	for i in range(len(L)-1):
		den = np.sum([(enh_ct > L[i]) & (enh_ct <= L[i+1])])
		if den == 0 or (exclude_small and den <= 25):
			lips.append(np.nan)
		else:
			lips.append(np.sum([(resp_ct > L[i]) & (resp_ct <= L[i+1])]) / den)

	return lips

def get_vol_coverage(lesion_id, target_dir):
	#tumor,_ = masks.get_mask(P['mrbl']['tumor'])
	#tumor = tumor/tumor.max()
	#mask,_ = masks.get_mask(P['mrbl']['enh'])
	#mask = mask/mask.max()
	#row.append(mask.sum()/tumor.sum()) #enhancing_vol%
	ret = []
	P = lm.get_paths_dict(lesion_id, target_dir)

	if exists(P['ct24']['midlip'] + ".off"):
		img, D = hf.nii_load(P['ct24']['img'])
		tumor,_ = masks.get_mask(P['ct24']['tumor'], D, img.shape)
		tumor = tumor/tumor.max()

		mask,_ = masks.get_mask(P['ct24']['midlip'], D, img.shape)
		mask = mask/mask.max()
		ret.append((mask*tumor).sum()/tumor.sum()) #lipcoverage_vol%

		if exists(P['ct24']['highlip'] + ".off"):
			mask,_ = masks.get_mask(P['ct24']['highlip'], D, img.shape)
			mask = mask/mask.max()
			ret.append((mask*tumor).sum()/tumor.sum()) #high_lip
		else:
			ret.append(0)
	else:
		ret = [0,0]

	return ret

def get_row_entry(lesion_id, target_dir, liplvls):
	import importlib
	importlib.reload(masks)
	importlib.reload(tr)

	P = lm.get_paths_dict(lesion_id, target_dir)

	row = []
	row += get_vol_coverage(lesion_id, target_dir)
	
	row.append(get_rim_coverage(lesion_id, target_dir, liplvls[1], r_frac=.85))
	row += get_peripheral_coverage(lesion_id, target_dir, liplvls[1:3], .15)

	return row

def get_actual_order(category, df, order):
	importlib.reload(lvis)
	if order is None:
		return None
	
	vals = np.unique(df.dropna(subset=[category])[category].values)
	new_order=[]
	for o in order:
		new_order.append([v for v in vals if v.startswith(o)][0])
		
	return new_order

###########################
### Features
###########################

def get_texture_feats(row, img):
	feats = mah.haralick(img, distance=2)
	contrast = feats[:,1].mean()
	var = feats[:,3].mean()
	idm = feats[:,4].mean()

	row += [contrast, var, idm]
	
	return row

def get_rim_coverage(lesion_id, target_dir, min_threshold, r_frac=.85):
	importlib.reload(hf)
	P = lm.get_paths_dict(lesion_id, target_dir)

	if not exists(P['ball']['ct24']['img']):
		return np.nan

	img, dims = hf.nii_load(P['ball']['ct24']['img'])
	M, _ = masks.get_mask(P['ball']['mask'])
	M = M/M.max()
	nonzeros = np.argwhere(M)
	R = (nonzeros[:,0].max() - nonzeros[:,0].min()) / 2

	core_M = hf.zeropad(ball(int(R*r_frac)), img.shape)
	core_I = np.sum(img[core_M != 0]) / core_M.sum()
	std_I = np.std(img[M != 0])

	threshold = max(min_threshold, core_I)
	
	M = M - hf.zeropad(ball(int(R*r_frac)), img.shape)
	img = (img*M - threshold)/10#np.ceil((img*M - threshold) / std_I)
	img[img < 0] = 0
	#img[img > 5] = 5

	return img.sum() / M.sum() 

def get_peripheral_coverage(lesion_id, target_dir, thresholds=[100,150], dR=.15):
	P = lm.get_paths_dict(lesion_id, target_dir)
	img, D = hf.nii_load(P['ct24']['img'])
	M = masks.get_mask(P['ct24']['tumor'], D, img.shape)[0]
	M = M/M.max()

	B4 = ball(4)
	B4 = B4[:,:,[1,3,5,6,8]]
	M = binary_dilation(M, B4) - M

	"""ball = masks.get_mask(P['ball']['mask'])[0]
				nonzeros = np.argwhere(ball)
				
				R = (nonzeros[:,0].max() - nonzeros[:,0].min()) / 2
				m = ball.shape[0]//2
			
				img = hf.nii_load(P['ball']['ct24']['img'])[0]
				img[img < 0] = 1
				img = img*(1-ball/ball.max())
				
				for x in range(ball.shape[0]):
					for y in range(ball.shape[1]):
						for z in range(ball.shape[2]):
							if img[x,y,z] == 0:
								continue
								
							X = m-.5-x
							Y = m-.5-y
							Z = m-.5-z
							r = (X*X+Y*Y+Z*Z)**.5
							if r > R*(1+dR):
								img[x,y,z] = 0"""

	return [(img*M > T).sum() / M.sum() for T in thresholds]

def get_qEASL(lesion_id, target_dir):
	P = lm.get_paths_dict(lesion_id, target_dir)

	Abl, Dbl = hf.nii_load(P['mrbl']['art'])
	A30, D30 = hf.nii_load(P['mr30']['art'])
	return (masks.get_mask(P['mr30']['enh'], D30, A30.shape)[0].sum() * np.product(D30)) / \
			(masks.get_mask(P['mrbl']['enh'], Dbl, Abl.shape)[0].sum() * np.product(Dbl)) - 1

###########################
### Etc.
###########################

def get_best_T_lip(lesion_id, target_dir, T_lip=150):
	importlib.reload(lm)
	P = lm.get_paths_dict(lesion_id, target_dir)
	ct = hf.nii_load(P['ct24Tx']['crop']['img'])[0]
	M = masks.get_mask(P['ct24Tx']['crop']['tumor'])[0]
	ct[M == 0] = np.nan
	ct_U = ct >= T_lip
	ct_L = ct < T_lip
	art = hf.nii_load(P['ct24Tx']['mrbl']['art'])[0].astype(int)
	sub = hf.nii_load(P['ct24Tx']['mrbl']['sub'])[0].astype(int)
	
	dice_best_art = 0
	for T_art in range(art[M != 0].min(), art[M != 0].max()+1):
		intersec = ct_U[art >= T_art].sum() + ct_L[art < T_art].sum()
		if intersec > dice_best_art:
			T_best_art = T_art
			dice_best_art = intersec
	dice_best_art /= (M>0).sum()

	dice_best_sub = 0
	for T_sub in range(sub[M != 0].min(), sub[M != 0].max()+1):
		intersec = ct_U[sub >= T_sub].sum() + ct_L[sub < T_sub].sum()
		if intersec > dice_best_sub:
			T_best_sub = T_sub
			dice_best_sub = intersec
	dice_best_sub /= (M>0).sum()

	return T_best_art, dice_best_art, T_best_sub, dice_best_sub
