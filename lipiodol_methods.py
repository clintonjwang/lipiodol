import config
import copy
import importlib
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
### Preprocessing
###########################

def check_multi_tumors(lesion_id, target_dir):
	P = get_paths_dict(lesion_id, target_dir)
	
	mask_path = P['mrbl']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(lesion_id, "has multiple tumors on BL MR")
		
	mask_path = P['mr30']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(lesion_id, "has multiple tumors on 30d MR")
		
	mask_path = P['ct24']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(lesion_id, "has multiple tumors on 24h CT")

def restrict_masks(lesion_id, target_dir):
	P = get_paths_dict(lesion_id, target_dir)
	
	mask_path = P['mrbl']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:# and tumor_vols[0] < 10*tumor_vols[1]:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=P['mrbl']['art'])
		
	mask_path = P['mr30']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=P['mr30']['art'])
		
	mask_path = P['ct24']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=P['ct24']['img'])

def spherize(lesion_id, target_dir, R=1.):
	importlib.reload(reg)
	def ball_ct_batch():
		reg.transform_region(P['ct24']['img'], xform_path, crops, pads, [R]*3, P['ball']['ct24']['img'],
								 intermed_shape=ball_shape)

		if exists(P['ct24']['midlip']+".off"):
			reg.transform_mask(P['ct24']['midlip'], P['ct24']['img'], xform_path,
								 crops, pads, [R]*3, P['ball']['ct24']['midlip'], intermed_shape=ball_shape)
			if exists(P['ct24']['highlip']+".off"):
				reg.transform_mask(P['ct24']['highlip'], P['ct24']['img'], xform_path,
									 crops, pads, [R]*3, P['ball']['ct24']['highlip'], intermed_shape=ball_shape)

	def ball_mr_batch(mod):
		reg.transform_region(P[mod]['art'], xform_path, crops, pads, [R]*3, P['ball'][mod]['art'], intermed_shape=ball_shape)
		
		if exists(P['ball'][mod]['enh']+".off"):
			reg.transform_mask(P[mod]['enh'], P[mod]['art'], xform_path,
								 crops, pads, [R]*3, P['ball'][mod]['enh'], intermed_shape=ball_shape)

	P = get_paths_dict(lesion_id, target_dir)
	
	ctmask,ctd = masks.get_mask(P['ct24']['tumor'], img_path=P['ct24']['img'])
	mrmask,mrd = masks.get_mask(P['mrbl']['tumor'], img_path=P['mrbl']['art'])
	ctmask = hf.crop_nonzero(ctmask)[0]
	mrmask = hf.crop_nonzero(mrmask)[0]
	CT = np.max([ctmask.shape[i] * ctd[i] for i in range(3)])
	MRBL = np.max([mrmask.shape[i] * mrd[i] for i in range(3)])
	
	mrmask,mrd = masks.get_mask(P['mr30']['tumor'], img_path=P['mr30']['art'])
	mrmask = hf.crop_nonzero(mrmask)[0]
	MR30 = np.max([mrmask.shape[i] * mrd[i] for i in range(3)])
	
	if CT > MRBL and CT > MR30:
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['ct24']['img'], P['ct24']['tumor'], mask_path=P['ball']['mask'])
		ball_shape = masks.get_mask(P['ball']['mask'])[0].shape
		ball_ct_batch()

		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mrbl']['art'], P['mrbl']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mrbl')

		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mr30']['art'], P['mr30']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mr30')
		
	elif MRBL > MR30:
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mrbl']['art'],
											P['mrbl']['tumor'], mask_path=P['ball']['mask'])
		ball_shape = masks.get_mask(P['ball']['mask'])[0].shape
		ball_mr_batch('mrbl')
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['ct24']['img'], P['ct24']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_ct_batch()

		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mr30']['art'], P['mr30']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mr30')
		
	else:
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mr30']['art'], P['mr30']['tumor'], mask_path=P['ball']['mask'])
		ball_shape = masks.get_mask(P['ball']['mask'])[0].shape
		ball_mr_batch('mr30')
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mrbl']['art'], P['mrbl']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mrbl')
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['ct24']['img'], P['ct24']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_ct_batch()

def reg_to_ct24(lesion_id, target_dir, D=[1.,1.,2.5], padding=.2):
	importlib.reload(reg)
	P = get_paths_dict(lesion_id, target_dir)

	ct24, ct24_dims = hf.nii_load(P['ct24']['img'])
	fmod='ct24'

	mod = 'mrbl'
	xform_path, crops, pad_m = reg.get_mask_Tx(P[fmod]['img'], P[fmod]['tumor'],
					P[mod]['art'], P[mod]['tumor'], padding=padding, D=D)
	
	crop_ct24 = hf.crop_nonzero(ct24, crops[1])[0]
	t_shape = crop_ct24.shape
	reg.transform_region(P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['art'], target_shape=t_shape, D=D);
	if exists(P[mod]['enh']+".off"):
		reg.transform_mask(P[mod]['enh'], P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['enh'], target_shape=t_shape, D=D);

	hf.save_nii(crop_ct24, P['ct24Tx']['crop']['img'], ct24_dims)
	M = masks.get_mask(P['ct24']['tumor'], ct24_dims, ct24.shape)[0]
	M = hf.crop_nonzero(M, crops[1])[0]
	masks.save_mask(M, P['ct24Tx']['crop']['tumor'], ct24_dims)

	mod = 'mr30'
	xform_path, crops, pad_m = reg.get_mask_Tx(P[fmod]['img'], P[fmod]['tumor'],
					P[mod]['art'], P[mod]['tumor'], padding=padding, D=D)
	
	reg.transform_region(P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['art'], target_shape=t_shape, D=D);
	if exists(P[mod]['enh']+".off"):
		reg.transform_mask(P[mod]['enh'], P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['enh'], target_shape=t_shape, D=D);
	
###########################
### Assess correlations
###########################

"""def enhancing_to_nec(lesion_id, target_dir, liplvls=[0,75,160,215]):
	P = get_paths_dict(lesion_id, target_dir)

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
	P = get_paths_dict(lesion_id, target_dir)
	L = liplvls + [10000]

	M = masks.get_mask(P['ct24Tx']['crop']['tumor'])[0]
	if exists(P['ct24Tx']['mrbl']['enh'] + ".off"):
		mrbl_enh = masks.get_mask(P['ct24Tx']['mrbl']['enh'])[0]
		mrbl_enh = mrbl_enh/mrbl_enh.max()
	else:
		mrbl_enh = np.zeros(M.shape)
	mrbl_nec = M/M.max()-mrbl_enh
	mrbl_nec[mrbl_nec < 0] = 0

	ct24 = hf.nii_load(P['ct24Tx']['crop']['img'])[0]
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

def lip_to_response(lesion_id, target_dir, liplvls, exclude_small=True):
	P = get_paths_dict(lesion_id, target_dir)
	L = liplvls + [10000]

	M = masks.get_mask(P['ct24Tx']['crop']['tumor'])[0]
	M = M/M.max()
	if exists(P['ct24Tx']['mrbl']['enh'] + ".off"):
		mrbl_enh = masks.get_mask(P['ct24Tx']['mrbl']['enh'])[0]
		mrbl_enh = mrbl_enh/mrbl_enh.max()
	else:
		return [np.nan]*len(liplvls)

	if exists(P['ct24Tx']['mr30']['enh'] + ".off"):
		mr30d_enh = masks.get_mask(P['ct24Tx']['mr30']['enh'])[0]
		mr30d_nec = M * (1 - mr30d_enh/mr30d_enh.max())
	else:
		mr30d_enh = np.zeros(M.shape)
		mr30d_nec = M

	ct24 = hf.nii_load(P['ct24Tx']['crop']['img'])[0]
	ct24[ct24 < 0] = 1

	enh_ct = ct24 * mrbl_enh
	V = np.sum(enh_ct != 0)
	resp = mrbl_enh * mr30d_nec
	resp_ct = ct24 * resp

	lips = []
	B3 = ball(3)
	B3 = B3[:,:,[0,2,3,4,6]]
	for i in range(len(L)-1):
		lip_segment = (enh_ct <= L[i]) | (enh_ct > L[i+1])
		lip_segment = binary_closing(binary_opening(lip_segment, B3))

		den = np.sum([(enh_ct > L[i]) & (enh_ct <= L[i+1])])
		if den == 0 or (exclude_small and den <= 50):#(den / V <= .05 or den <= 50)):
			lips.append(np.nan)
		else:
			lips.append(np.sum([(resp_ct > L[i]) & (resp_ct <= L[i+1])]) / den)

	return lips

def vascular_to_deposition_ball(lesion_id, target_dir, liplvls, exclude_small=True):
	P = get_paths_dict(lesion_id, target_dir)
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
	paths = get_paths(lesion_id, target_dir, check_valid=False)
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
	P = get_paths_dict(lesion_id, target_dir)

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

	P = get_paths_dict(lesion_id, target_dir)

	row = []
	row += get_vol_coverage(lesion_id, target_dir)
	
	#ball_IV = get_avg_ball_intensity(P['ball']['ct24']['img'], ball_mask_path)
	#row = get_rim_coverage(row, masks.get_mask(P['ball']['mrbl']['enh'])[0] + 1, ball_mask_path, 1.5)
	row.append(get_rim_coverage(lesion_id, target_dir, liplvls[1], r_frac=.85))
	row += get_peripheral_coverage(lesion_id, target_dir, liplvls[1:3], .15)

	return row

def get_actual_order(category, df, order):
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

def get_rim_coverage(lesion_id, target_dir, min_threshold=100, r_frac=.85):
	importlib.reload(hf)
	P = get_paths_dict(lesion_id, target_dir)

	if not exists(P['ball']['ct24']['img']):
		return np.nan

	img, dims = hf.nii_load(P['ball']['ct24']['img'])
	M, _ = masks.get_mask(P['ball']['mask'])
	nonzeros = np.argwhere(M)
	R = (nonzeros[:,0].max() - nonzeros[:,0].min()) / 2

	core_M = hf.zeropad(ball(int(R*r_frac)), img.shape)
	core_I = np.sum(img[core_M != 0]) / core_M.sum()
	std_I = np.std(img[M != 0])

	threshold = max(min_threshold, core_I)
	
	M = M/M.max() - hf.zeropad(ball(int(R*r_frac)), img.shape)
	img = np.ceil((img*M - threshold) / std_I)
	img[img < 0] = 0

	return img.sum() / M.sum() 

def get_peripheral_coverage(lesion_id, target_dir, thresholds=[100,150], dR=.15):
	P = get_paths_dict(lesion_id, target_dir)
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
	P = get_paths_dict(lesion_id, target_dir)

	Abl, Dbl = hf.nii_load(P['mrbl']['art'])
	A30, D30 = hf.nii_load(P['mr30']['art'])
	return (masks.get_mask(P['mr30']['enh'], D30, A30.shape)[0].sum() * np.product(D30)) / \
			(masks.get_mask(P['mrbl']['enh'], Dbl, Abl.shape)[0].sum() * np.product(Dbl)) - 1

############## Spherical shell analysis

def calc_intensity_shells_angles(img, ball_mask_path):
	IVs = np.zeros((181,361,7)) # avg voxel intensity at core (0-50%), 50-60%, 60-70%, 70-80%, 80-90%, 90-100%, 100%-120%, 120%-150%
	contributions = np.ones((181,361,7)) * 1e-4
	
	ball, _ = masks.get_mask(ball_mask_path)
	nonzeros = np.argwhere(ball)
	
	R = (nonzeros[:,0].max() - nonzeros[:,0].min()) / 2
	m = ball.shape[0]//2
	
	for x in range(ball.shape[0]):
		for y in range(ball.shape[1]):
			for z in range(ball.shape[2]):
					
				X = m-.5-x
				Y = m-.5-y
				Z = m-.5-z
				r = (X*X+Y*Y+Z*Z)**.5
				if r > R*1.5 or r < R*.5:
					continue
					
				theta = degrees(math.acos(Z/r))
				phi = degrees(math.atan2(Y,X)+pi)
				
				if r >= R*1.2:
					shell_num = -1
				elif r > R*1:
					shell_num = -2
				else:
					shell_num = int((r/R-.5)*10)
					
				dt = abs(round(theta) - theta)
				dp = abs(round(phi) - theta)
				IVs[round(theta), round(phi), shell_num] += (dt+dp)/2 * img[x,y,z]
				contributions[round(theta), round(phi), shell_num] += (dt+dp)/2
				
				if False:
					dt = math.ceil(theta) - theta
					dp = math.ceil(phi) - theta
					IVs[math.ceil(theta), math.ceil(phi), shell_num] += (dt+dp)/2 * img[x,y,z]
					contributions[math.ceil(theta), math.ceil(phi), shell_num] += (dt+dp)/2

					IVs[math.floor(theta), math.ceil(phi), shell_num] += (1-dt+dp)/2 * img[x,y,z]
					contributions[math.floor(theta), math.ceil(phi), shell_num] += (1-dt+dp)/2

					IVs[math.ceil(theta), math.floor(phi), shell_num] += (dt+1-dp)/2 * img[x,y,z]
					contributions[math.ceil(theta), math.floor(phi), shell_num] += (dt+1-dp)/2

					IVs[math.floor(theta), math.floor(phi), shell_num] += (1-dt+1-dp)/2 * img[x,y,z]
					contributions[math.floor(theta), math.floor(phi), shell_num] += (1-dt+1-dp)/2
	
	return IVs / contributions

def fibonacci_sphere(samples=1, spherical_coords=False, randomize=False):
	"""Even sampling of points on the sphere"""
	rnd = 1.
	if randomize:
		rnd = random.random() * samples
		
	points = []
	offset = 2./samples
	increment = math.pi * (3. - math.sqrt(5.))

	for i in range(samples):
		y = ((i * offset) - 1) + (offset / 2);
		r = math.sqrt(1 - pow(y,2))

		phi = ((i + rnd) % samples) * increment
		
		if spherical_coords:
			x = math.cos(phi) * r
			z = math.sin(phi) * r
			
			theta = degrees(math.acos(z/r))
			phi = degrees(math.atan2(y,x)+pi)

			points.append([theta,phi,1])
			
		else:
			x = math.cos(phi) * r
			z = math.sin(phi) * r

			points.append([x,y,z])

	return np.array(points)

def get_avg_ball_intensity(reg_img_path, ball_mask_path):
	img, dims = hf.nii_load(reg_img_path)
	ball, _ = masks.get_mask(ball_mask_path)
	nonzeros = np.argwhere(ball)
	
	R = (nonzeros[:,0].max() - nonzeros[:,0].min()) / 2
	m = ball.shape[0]//2
	img[ball == 0] = np.nan
	V = ball.sum()
	I = np.nanmedian(img)
	return I#/V


###########################
### File I/O
###########################

def check_paths(lesion_id, target_dir):
	P = get_paths_dict(lesion_id, target_dir)
	for path in [P['mask'], P['nii'], P['ct24']['img'], P['ct24']['tumor'], P['ct24']['liver'], \
			P['mrbl']['art'], P['mrbl']['pre'], P['mrbl']['sub'], \
			P['mrbl']['tumor'], P['mrbl']['liver'], \
			P['mr30']['art'], P['mr30']['pre'], \
			P['mr30']['tumor'], P['mr30']['liver']]:
		if not exists(path) and not exists(path+".ics"):
			print(path, "does not exist!")
			raise ValueError(path)

	#if P['mrbl']['enh'] and P['mrbl']['nec']:
	#if P['mr30']['enh'] and P['mr30']['nec']:

	P['ball']['ct24']['img'] = join(nii_dir, "ct24_ball.nii")
	ball_mribl_path = join(nii_dir, "mribl_ball.nii")
	ball_mri30d_path = join(nii_dir, "mri30d_ball.nii")
	ball_mask_path = join(mask_dir, "ball_mask")
	P['ball']['mrbl']['enh'] = join(mask_dir, "ball_mribl_enh_mask")
	P['ball']['mr30']['enh'] = join(mask_dir, "ball_mri30d_enh_mask")

	midlip_mask_path = join(mask_dir, "mid_lipiodol")
	ball_midlip_mask_path = join(mask_dir, "ball_mid_lipiodol")
	highlip_mask_path = join(mask_dir, "high_lipiodol")
	ball_highlip_mask_path = join(mask_dir, "ball_lipiodol")

	paths += [P['ball']['ct24']['img'], ball_mribl_path, ball_mri30d_path, \
			ball_mask_path, P['ball']['mrbl']['enh'], P['ball']['mr30']['enh'], \
			midlip_mask_path, ball_midlip_mask_path, \
			highlip_mask_path, ball_highlip_mask_path]

	#if flag:
	#	paths[-1] = None
	return paths

def get_paths_dict(lesion_id, target_dir):
	if not exists(join(target_dir, lesion_id)):
		raise ValueError(lesion_id, "does not exist!")

	mask_dir = join(target_dir, lesion_id, "masks")
	nii_dir = join(target_dir, lesion_id, "nii_files")
	if not exists(nii_dir):
		os.makedirs(nii_dir)

	P = {'mask':mask_dir, 'nii':nii_dir, 'mrbl':{}, 'ct24':{}, 'mr30':{},
		'ball':{'mrbl':{}, 'ct24':{}, 'mr30':{}}, 'mr30Tx':{'mrbl':{}, 'ct24':{}},
			'ct24Tx':{'mrbl':{}, 'crop':{}, 'mr30':{}}}

	P['ct24']['img'] = join(target_dir, lesion_id, "nii_files", "ct24.nii.gz")
	P['ct24']['tumor'] = glob.glob(join(mask_dir, "tumor*24h*.ids"))
	P['ct24']['liver'] = glob.glob(join(mask_dir, "wholeliver_24hCT*.ids"))
	P['ct24']['lowlip'] = join(mask_dir, "lipiodol_low")
	P['ct24']['midlip'] = join(mask_dir, "lipiodol_mid")
	P['ct24']['highlip'] = join(mask_dir, "lipiodol_high")
	if len(P['ct24']['tumor']) > 0:
		P['ct24']['tumor'] = P['ct24']['tumor'][0]
	else:
		raise ValueError('tumor')
	if len(P['ct24']['liver']) > 0:
		P['ct24']['liver'] = P['ct24']['liver'][0]
	else:
		raise ValueError('liver')

	P['mrbl']['art'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_art.nii.gz")
	P['mrbl']['pre'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_pre.nii.gz")
	P['mrbl']['sub'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_sub.nii.gz")
	P['mrbl']['equ'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_equ.nii.gz")
	P['mr30']['art'] = join(target_dir, lesion_id, "MRI-30d", "mr30_art.nii.gz")
	P['mr30']['pre'] = join(target_dir, lesion_id, "MRI-30d", "mr30_pre.nii.gz")
	P['mr30']['sub'] = join(target_dir, lesion_id, "MRI-30d", "mr30_sub.nii.gz")
	P['mr30']['equ'] = join(target_dir, lesion_id, "MRI-30d", "mr30_equ.nii.gz")

	P['mrbl']['tumor'] = join(mask_dir, "tumor_BL_MRI")
	P['mrbl']['liver'] = join(mask_dir, "mribl_liver")
	P['mrbl']['enh'] = join(mask_dir, "enh_bl")
	P['mrbl']['nec'] = join(mask_dir, "nec_bl")
	P['mr30']['tumor'] = join(mask_dir, "tumor_30dFU_MRI")
	P['mr30']['liver'] = join(mask_dir, "mri30d_liver")
	P['mr30']['enh'] = join(mask_dir, "enh_30d")
	P['mr30']['nec'] = join(mask_dir, "nec_30d")


	if not exists(join(nii_dir, "reg")):
		os.makedirs(join(nii_dir, "reg"))
	if not exists(join(mask_dir, "reg")):
		os.makedirs(join(mask_dir, "reg"))

	P['ball']['mask'] = join(mask_dir, "reg", "ball_mask")
	P['ball']['mrbl']['art'] = join(nii_dir, "reg", "ball_mrbl.nii")
	P['ball']['mrbl']['enh'] = join(mask_dir, "reg", "ball_mrbl_enh_mask")
	P['ball']['mr30']['art'] = join(nii_dir, "reg", "ball_mr30.nii")
	P['ball']['mr30']['enh'] = join(mask_dir, "reg", "ball_mri30d_enh_mask")
	P['ball']['ct24']['img'] = join(nii_dir, "reg", "ball_ct24.nii")
	P['ball']['ct24']['lowlip'] = join(mask_dir, "reg", "ball_lowlip")
	P['ball']['ct24']['midlip'] = join(mask_dir, "reg", "ball_midlip")
	P['ball']['ct24']['highlip'] = join(mask_dir, "reg", "ball_highlip")

	P['mr30Tx']['mrbl']['art'] = join(nii_dir, "reg", "mr30Tx-mrbl-art.nii")
	P['mr30Tx']['mrbl']['enh'] = join(mask_dir, "reg", "mr30Tx-mrbl-enh")
	P['mr30Tx']['ct24']['img'] = join(nii_dir, "reg", "mr30Tx-ct24-img.nii")
	P['mr30Tx']['ct24']['midlip'] = join(mask_dir, "reg", "mr30Tx-ct24-midlip")
	P['mr30Tx']['ct24']['highlip'] = join(mask_dir, "reg", "mr30Tx-ct24-highlip")

	P['ct24Tx']['mrbl']['art'] = join(nii_dir, "reg", "ct24Tx-mrbl-art.nii")
	P['ct24Tx']['mrbl']['enh'] = join(mask_dir, "reg", "ct24Tx-mrbl-enh")
	P['ct24Tx']['mr30']['art'] = join(nii_dir, "reg", "ct24Tx-mr30-art.nii")
	P['ct24Tx']['mr30']['enh'] = join(mask_dir, "reg", "ct24Tx-mr30-enh")
	P['ct24Tx']['crop']['img'] = join(nii_dir, "reg", "ct24Tx-crop-img.nii")
	P['ct24Tx']['crop']['tumor'] = join(mask_dir, "reg", "ct24Tx-crop-tumor")
	P['ct24Tx']['crop']['midlip'] = join(mask_dir, "reg", "ct24Tx-crop-midlip")
	P['ct24Tx']['crop']['highlip'] = join(mask_dir, "reg", "ct24Tx-crop-highlip")

	return P

###########################
### Segmentation methods
###########################

def seg_lipiodol(lesion_id, target_dir, thresholds=[75,160,215]):
	P = get_paths_dict(lesion_id, target_dir)
	
	img, dims = hf.nii_load(P['ct24']['img'])

	low_mask = copy.deepcopy(img)
	low_mask = low_mask > thresholds[0]
	low_mask = binary_closing(binary_opening(low_mask, structure=np.ones((2,2,1))), structure=np.ones((2,2,1)))
	mid_mask = copy.deepcopy(img)
	mid_mask = mid_mask > thresholds[1]
	mid_mask = binary_closing(mid_mask)
	high_mask = copy.deepcopy(img)
	high_mask = high_mask > thresholds[2]

	mask,_ = masks.get_mask(P['ct24']['liver'], dims, img.shape)
	low_mask = low_mask*mask
	mid_mask = mid_mask*mask
	high_mask = high_mask*mask
	
	masks.save_mask(low_mask, P['ct24']['lowlip'], dims, save_mesh=True)
	masks.save_mask(mid_mask, P['ct24']['midlip'], dims, save_mesh=True)
	masks.save_mask(high_mask, P['ct24']['highlip'], dims, save_mesh=True)

def seg_target_lipiodol(img, save_folder, ct_dims, threshold=150, num_tumors=1):
	mid_mask = copy.deepcopy(img)
	mid_mask = mid_mask > threshold
	mid_mask = binary_closing(mid_mask)
	
	mask1 = copy.deepcopy(img)
	mask1 = mask1 > threshold
	B2 = ball(2)
	B2 = B2[:,:,[0,2,4]]
	mask1 = binary_dilation(mask1, np.ones((3,3,1)))
	
	if num_tumors == 1:
		tumor_labels, num_labels = label(mask1, return_num=True)
		label_sizes = [np.sum(tumor_labels == label_id) for label_id in range(1,num_labels+1)]
		biggest_label = label_sizes.index(max(label_sizes))+1
		mask1[tumor_labels != biggest_label] = 0
	
	mask1 = binary_opening(binary_closing(mask1, structure=B2, iterations=2), structure=B2, iterations=2)
	
	if num_tumors == 1:
		tumor_labels, num_labels = label(mask1, return_num=True)
		label_sizes = [np.sum(tumor_labels == label_id) for label_id in range(1,num_labels+1)]
		biggest_label = label_sizes.index(max(label_sizes))+1
		mask1[tumor_labels != biggest_label] = 0
	
	target_mask = mask1 * mid_mask
	nontarget_mask = (1-mask1) * mid_mask
	
	masks.save_mask(mask1, join(save_folder, "tumor_pred"), ct_dims, save_mesh=True)
	masks.save_mask(target_mask, join(save_folder, "target_lip"), ct_dims, save_mesh=True)
	masks.save_mask(nontarget_mask, join(save_folder, "nontarget_lip"), ct_dims, save_mesh=True)

def seg_liver_mri_from_path(mri_path, save_path, model, tumor_mask_path=None):
	mri_img, mri_dims = hf.nii_load(mri_path)
	seg_liver_mri(mri_img, save_path, mri_dims, model, tumor_mask_path)

def seg_liver_mri(mri_img, save_path, mri_dims, model, tumor_mask_path=None):
	"""Use a UNet to segment liver on MRI"""

	C = config.Config()
	#correct bias field!

	orig_shape = mri_img.shape

	x = mri_img
	x -= np.amin(x)
	x /= np.std(x)

	crops = list(map(int,[.05 * x.shape[0], .95 * x.shape[0]] + \
					[.05 * x.shape[1], .95 * x.shape[1]] + \
					[.05 * x.shape[2], .95 * x.shape[2]]))
	
	x = x[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]
	scale_shape = x.shape
	x = tr.rescale_img(x, C.dims)
	
	y = model.predict(np.expand_dims(x,0))[0]
	liver_mask = (y[:,:,:,1] > y[:,:,:,0]).astype(float)
	liver_mask = tr.rescale_img(liver_mask, scale_shape)#orig_shape)
	liver_mask = np.pad(liver_mask, ((crops[0], orig_shape[0]-crops[1]),
									 (crops[2], orig_shape[1]-crops[3]),
									 (crops[4], orig_shape[2]-crops[5])), 'constant')
	liver_mask = liver_mask > .5

	B3 = ball(3)
	B3 = B3[:,:,[0,2,3,4,6]]
	#B3 = ball(4)
	#B3 = B3[:,:,[1,3,5,6,8]]
	liver_mask = binary_opening(binary_closing(liver_mask, B3, 1), B3, 1)

	labels, num_labels = label(liver_mask, return_num=True)
	label_sizes = [np.sum(labels == label_id) for label_id in range(1,num_labels+1)]
	biggest_label = label_sizes.index(max(label_sizes))+1
	liver_mask[labels != biggest_label] = 0

	if tumor_mask_path is not None:
		tumor_mask, _ = masks.get_mask(tumor_mask_path, mri_dims, mri_img.shape)
		liver_mask[tumor_mask > tumor_mask.max()/2] = liver_mask.max()

	masks.save_mask(liver_mask, save_path, mri_dims, save_mesh=True)

def seg_liver_ct(ct_path, save_path, model, tumor_mask_path=None):
	"""Use a UNet to segment liver on CT"""

	C = config.Config()
	ct_img, ct_dims = hf.nii_load(ct_path)

	orig_shape = ct_img.shape

	x = tr.apply_window(ct_img)
	x -= np.amin(x)

	crops = list(map(int,[.05 * x.shape[0], .95 * x.shape[0]] + \
					[.05 * x.shape[1], .95 * x.shape[1]] + \
					[.05 * x.shape[2], .95 * x.shape[2]]))
	
	x = x[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]
	scale_shape = x.shape
	x = tr.rescale_img(x, C.dims)
	
	y = model.predict(np.expand_dims(x,0))[0]
	liver_mask = (y[:,:,:,1] > y[:,:,:,0]).astype(float)
	liver_mask[x < 30] = 0
	liver_mask = tr.rescale_img(liver_mask, scale_shape)
	liver_mask = np.pad(liver_mask, ((crops[0], orig_shape[0]-crops[1]),
									 (crops[2], orig_shape[1]-crops[3]),
									 (crops[4], orig_shape[2]-crops[5])), 'constant')

	B3 = ball(3)
	B3 = B3[:,:,[0,2,3,4,6]]
	liver_mask = binary_opening(binary_closing(liver_mask, B3, 1), B3, 1)

	labels, num_labels = label(liver_mask, return_num=True)
	label_sizes = [np.sum(labels == label_id) for label_id in range(1,num_labels+1)]
	biggest_label = label_sizes.index(max(label_sizes))+1
	liver_mask[labels != biggest_label] = 0

	if tumor_mask_path is not None:
		tumor_mask, _ = masks.get_mask(tumor_mask_path, ct_dims, ct_img.shape)
		liver_mask[tumor_mask > tumor_mask.max()/2] = liver_mask.max()
	
	masks.save_mask(liver_mask, save_path, ct_dims, save_mesh=True)