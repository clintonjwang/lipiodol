import config
import copy
import mahotas.features as mah
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
import os
from os.path import *
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation
from skimage.morphology import ball, label

###########################
### Preprocessing
###########################

def check_multi_tumors(patient_id, target_dir):
	paths = get_paths(patient_id, target_dir, check_valid=False)

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths
	
	mask_path = mribl_tumor_mask_path
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(patient_id, "has multiple tumors on BL MR")
		
	mask_path = mri30d_tumor_mask_path
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(patient_id, "has multiple tumors on 30d MR")
		
	mask_path = ct24_tumor_mask_path
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(patient_id, "has multiple tumors on 24h CT")

def restrict_masks(patient_id, target_dir):
	paths = get_paths(patient_id, target_dir)

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths
	
	mask_path = mribl_tumor_mask_path
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:# and tumor_vols[0] < 10*tumor_vols[1]:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=mribl_art_path)
		
	mask_path = mri30d_tumor_mask_path
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=mri30d_art_path)
		
	mask_path = ct24_tumor_mask_path
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=ct24_path)

def spherize(patient_id, target_dir):
	import importlib
	importlib.reload(reg)
	def ball_ct_batch():
		_ = reg.transform_region(ct24_path, xform_path, crops, pads, [1.]*3, ball_ct24_path,
								 mask_scale=mask_scale, target_shape=target_shape)
		try:
			_ = reg.transform_mask(highlip_mask_path, ct24_path, xform_path,
								 crops, pads, [1.]*3, ball_highlip_mask_path, mask_scale=mask_scale, target_shape=target_shape)
		except:
			print(ball_highlip_mask_path, "is empty")
			os.remove(ball_highlip_mask_path+".ics")
			os.remove(ball_highlip_mask_path+".ids")
		_ = reg.transform_mask(midlip_mask_path, ct24_path, xform_path,
							 crops, pads, [1.]*3, ball_midlip_mask_path, mask_scale=mask_scale, target_shape=target_shape)
		
	def ball_mrbl_batch():
		_ = reg.transform_region(mribl_art_path, xform_path, crops, pads, [1.]*3, ball_mribl_path, target_shape=target_shape)
		_ = reg.transform_mask(mribl_enh_mask_path, mribl_art_path, xform_path,
							 crops, pads, [1.]*3, ball_mribl_enh_mask_path, target_shape=target_shape)
		
	def ball_mr30_batch():
		_ = reg.transform_region(mri30d_art_path, xform_path, crops, pads, [1.]*3, ball_mri30d_path, target_shape=target_shape)
		_ = reg.transform_mask(mri30d_enh_mask_path, mri30d_art_path, xform_path,
							 crops, pads, [1.]*3, ball_mri30d_enh_mask_path, target_shape=target_shape)

	paths = get_paths(patient_id, target_dir)

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths
	
	ctmask,ctd = masks.get_mask(ct24_tumor_mask_path, img_path=ct24_path)
	mrmask,mrd = masks.get_mask(mribl_tumor_mask_path, img_path=mribl_art_path)
	ctmask = hf.crop_nonzero(ctmask)[0]
	mrmask = hf.crop_nonzero(mrmask)[0]
	mask_scale = (ctmask.sum()*np.product(ctd) / (mrmask.sum()*np.product(mrd)))**(1/6)
	CT = np.max([ctmask.shape[i] * ctd[i] / mask_scale for i in range(3)])
	MRBL = np.max([mrmask.shape[i] * mrd[i] for i in range(3)])
	
	mrmask,mrd = masks.get_mask(mri30d_tumor_mask_path, img_path=mri30d_art_path)
	mrmask = hf.crop_nonzero(mrmask)[0]
	MR30 = np.max([mrmask.shape[i] * mrd[i] for i in range(3)])
	
	if CT > MRBL and CT > MR30:
		xform_path, crops, pads = reg.get_mask_Tx_shape(ct24_path, ct24_tumor_mask_path, mask_scale, mask_path=ball_mask_path)
		target_shape = masks.get_mask(ball_mask_path)[0].shape
		ball_ct_batch()

		xform_path, crops, pads = reg.get_mask_Tx_shape(mribl_art_path, mribl_tumor_mask_path, ball_mask_path=ball_mask_path)
		ball_mrbl_batch()

		xform_path, crops, pads = reg.get_mask_Tx_shape(mri30d_art_path, mri30d_tumor_mask_path, ball_mask_path=ball_mask_path)
		ball_mr30_batch()
		
	elif MRBL > MR30:
		xform_path, crops, pads = reg.get_mask_Tx_shape(mribl_art_path,
											mribl_tumor_mask_path, mask_path=ball_mask_path)
		target_shape = masks.get_mask(ball_mask_path)[0].shape
		ball_mrbl_batch()
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(ct24_path, ct24_tumor_mask_path,
													mask_scale, ball_mask_path=ball_mask_path)
		ball_ct_batch()

		xform_path, crops, pads = reg.get_mask_Tx_shape(mri30d_art_path, mri30d_tumor_mask_path, ball_mask_path=ball_mask_path)
		ball_mr30_batch()
		
	else:
		xform_path, crops, pads = reg.get_mask_Tx_shape(mri30d_art_path, mri30d_tumor_mask_path, mask_path=ball_mask_path)
		target_shape = masks.get_mask(ball_mask_path)[0].shape
		ball_mr30_batch()
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(mribl_art_path, mribl_tumor_mask_path, ball_mask_path=ball_mask_path)
		ball_mrbl_batch()
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(ct24_path, ct24_tumor_mask_path,
													mask_scale, ball_mask_path=ball_mask_path)
		ball_ct_batch()

###########################
### Assess correlations
###########################

def enhancing_to_nec(patient_id, target_dir, liplvls=[0,100,150,200]):
	paths = get_paths(patient_id, target_dir)

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths

	mrbl_enh = masks.get_mask(ball_mribl_enh_mask_path)[0]
	mrbl_enh = mrbl_enh/mrbl_enh.max()
	mr30d_enh = masks.get_mask(ball_mri30d_enh_mask_path)[0]
	mr30d_enh = mr30d_enh/mr30d_enh.max()
	mrbl_nec = masks.difference(ball_mask_path, ball_mribl_enh_mask_path)
	mrbl_nec = mrbl_nec/mrbl_nec.max()
	mr30d_nec = masks.difference(ball_mask_path, ball_mri30d_enh_mask_path)
	mr30d_nec = mr30d_nec/mr30d_nec.max()
	
	ct24 = hf.nii_load(ball_ct24_path)[0]
	
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
	return lips

def lip_to_response(patient_id, target_dir, liplvls, exclude_small=False):
	paths = get_paths(patient_id, target_dir, check_valid=False)
	L = liplvls + [10000]

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths

	mrbl_enh = masks.get_mask(ball_mribl_enh_mask_path)[0]
	mrbl_enh = mrbl_enh/mrbl_enh.max()

	mr30d_nec = masks.difference(ball_mask_path, ball_mri30d_enh_mask_path)
	mr30d_nec = mr30d_nec/mr30d_nec.max()

	ct24 = hf.nii_load(ball_ct24_path)[0]
	resp = mrbl_enh * mr30d_nec

	enh_ct = ct24 * mrbl_enh
	V = np.sum(enh_ct > 0)

	resp_ct = ct24 * resp
	lips=[]
	for i in range(len(L)-1):
		den = np.sum([(enh_ct > L[i]) & (enh_ct <= L[i+1])])
		if den == 0 or (exclude_small and den / V <= .05):
			lips.append(np.nan)
		else:
			lips.append(np.sum([(resp_ct > L[i]) & (resp_ct <= L[i+1])]) / den)

	return lips

def vascular_to_deposition(patient_id, target_dir, liplvls, exclude_small=False):
	paths = get_paths(patient_id, target_dir, check_valid=False)
	L = liplvls + [10000]

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths

	mrbl_enh = masks.get_mask(ball_mribl_enh_mask_path)[0]
	ct24 = hf.nii_load(ball_ct24_path)[0]
	#lips = [np.sum([(ct24 * mrbl_enh/mrbl_enh.max()) > liplvls[i]]) / np.sum([ct24 > liplvls[i]]) for i in range(len(liplvls))]
	enh_ct = ct24 * mrbl_enh/mrbl_enh.max()
	ball = masks.get_mask(ball_mask_path)[0]
	ball_ct = ct24 * ball/ball.max()
	V = np.sum(ball_ct > 0)
	lips=[]
	for i in range(len(L)-1):
		den = np.sum([(ball_ct > L[i]) & (ball_ct <= L[i+1])])
		if den == 0 or (exclude_small and den <= 100): #den / V <= .05
			lips.append(np.nan)
		else:
			lips.append(np.sum([(enh_ct > L[i]) & (enh_ct <= L[i+1])]) / den)

	return lips

def get_vol_coverage(row, img, ball_mask_path, levels=[150,200]):
	ball_mask,_ = masks.get_mask(ball_mask_path)
	ball_mask = ball_mask/ball_mask.max()

	mask,_ = masks.get_mask(ball_mribl_enh_mask_path)
	mask = mask/mask.max()
	row.append(mask.sum()/ball_mask.sum()) #enhancing_vol%

	mask,_ = masks.get_mask(ball_midlip_mask_path)
	M = masks.intersection(ball_midlip_mask_path, ball_mask_path)
	M = M/M.max()
	row.append(M.sum()/ball_mask.sum()) #lipcoverage_vol%

	return row

def get_row_entry(patient_id, target_dir):
	import importlib
	importlib.reload(masks)
	importlib.reload(tr)

	paths = get_paths(patient_id, target_dir)

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths

	row = []
	#row = get_vol_coverage(row, )
	#row = get_vol_coverage(row)
	
	ball_IV = get_avg_ball_intensity(ball_ct24_path, ball_mask_path)
	#core_IV = get_avg_core_intensity(ball_ct24_path, ball_mask_path)
	row = get_rim_coverage(row, masks.get_mask(ball_mribl_enh_mask_path)[0] + 1, ball_mask_path, 1.5)
	row = get_rim_coverage(row, hf.nii_load(ball_ct24_path)[0], ball_mask_path, max(ball_IV,150))

	"""#mask = masks.get_mask(ball_mask_path)[0]

	img = hf.nii_load(ball_mribl_path)[0]
	img = img*mask/mask.max()
	img -= img[mask > 0].min()
	img = hf.crop_nonzero(img)[0]
	img = (img*255/img.max()).astype('uint8')
	row = get_texture_feats(row, img)

	img = hf.nii_load(ball_ct24_path)[0]
	img = img*mask/mask.max()
	img = tr.apply_window(img, limits=[0,300])
	img = hf.crop_nonzero(img)[0]
	img = (img*255/img.max()).astype('uint8')
	row = get_texture_feats(row, img)"""

	#row = get_peripheral_coverage(row, ball_ct24_path, ball_mask_path)

	return row


###########################
### Features
###########################


"""def get_rim_coverage(row, img, ball_mask_path, threshold):
	IVs = calc_intensity_shells_angles(img, ball_mask_path)
	IVs[IVs==0] = np.nan

	samples = fibonacci_sphere(3000, True, randomize=True)
	samples = np.round(samples).astype(int)
	s0 = samples[:,0]
	s1 = samples[:,1]
	#for i in range(IVs.shape[-1]):
	#    print(np.nanmean(IVs[s0,s1,i]))

	rim_percent = 0
	for i in range(5):
		num,den=0,0
		for j in range(len(s0)):
			if not np.isnan(IVs[s0[j],s1[j],i]):
				den += 1
				if IVs[s0[j],s1[j],i] > threshold:
					num += 1
		rim_percent = max([rim_percent, num/den])
	row.append(rim_percent)

	return row"""

def get_texture_feats(row, img):
	feats = mah.haralick(img, distance=2)
	contrast = feats[:,1].mean()
	var = feats[:,3].mean()
	idm = feats[:,4].mean()

	row += [contrast, var, idm]
	
	return row

def get_rim_coverage(row, img, ball_mask_path, threshold=100):
	ball = masks.get_mask(ball_mask_path)[0]
	nonzeros = np.argwhere(ball)
	
	R = (nonzeros[:,0].max() - nonzeros[:,0].min()) / 2
	m = ball.shape[0]//2

	img = img.astype(float)*ball/ball.max()
	img = [copy.deepcopy(img), copy.deepcopy(img), copy.deepcopy(img)]
	
	for x in range(ball.shape[0]):
		for y in range(ball.shape[1]):
			for z in range(ball.shape[2]):
				X = m-.5-x
				Y = m-.5-y
				Z = m-.5-z
				r = (X*X+Y*Y+Z*Z)**.5

				if img[0][x,y,z] != 0:
					if r < R*.9 or r > R:
						img[0][x,y,z] = 0

				if img[1][x,y,z] != 0:
					if r < R*.8 or r > R*.9:
						img[1][x,y,z] = 0

				#if img[2][x,y,z] != 0:
				#	if r < R*.7 or r > R*.8:
				#		img[2][x,y,z] = 0

	row.append(max( [(img[i] > threshold).sum() / (img[i] > 0).sum() for i in range(2)] ))

	return row

def get_peripheral_coverage(row, ball_ct24_path, ball_mask_path, threshold=100, dR=5):
	ball = masks.get_mask(ball_mask_path)[0]
	nonzeros = np.argwhere(ball)
	
	R = (nonzeros[:,0].max() - nonzeros[:,0].min()) / 2
	m = ball.shape[0]//2

	img = hf.nii_load(ball_ct24_path)[0]
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
				if r > R+dR:
					img[x,y,z] = 0

	row.append((img > threshold).sum() / (img > 0).sum())

	return row

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

def get_avg_core_intensity(reg_img_path, ball_mask_path, r_frac=.75):
	img, dims = hf.nii_load(reg_img_path)
	ball, _ = masks.get_mask(ball_mask_path)
	ball = ball/ball.max()
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
				if r > R*r_frac:
					ball[x,y,z] = 0
					
	img[ball == 0] = np.nan
	V = ball.sum()
	I = np.nansum(img)
	return I/V

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
### Visualization
###########################

def write_ranked_imgs(df, target_dir, column, img_type, root_dir, overwrite=False, mask_type=None, window=None):
	if not exists(root_dir):
		os.makedirs(root_dir)
		
	for ix,row in df.sort_values([column], ascending=False).iterrows():
		save_dir = join(root_dir, "%d_%s" % (row[column]*100, ix))
		
		patient_id = ix
		paths = get_paths(patient_id, target_dir, check_valid=False)

		mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
		mribl_art_path, mribl_pre_path, \
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


###########################
### File I/O
###########################

def get_paths(patient_id, target_dir, check_valid=True):
	if not exists(join(target_dir, patient_id)):
		raise ValueError(patient_id, "does not exist!")

	mask_dir = join(target_dir, patient_id, "masks")
	nii_dir = join(target_dir, patient_id, "nii_files")
	if not exists(nii_dir):
		os.makedirs(nii_dir)

	ct24_path = join(target_dir, patient_id, "nii_files", "ct24.nii.gz")
	ct24_tumor_mask_path = glob.glob(join(mask_dir, "tumor*24h*.ids"))
	ct24_liver_mask_path = glob.glob(join(mask_dir, "wholeliver_24hCT*.ids"))
	if len(ct24_tumor_mask_path) > 0:
		ct24_tumor_mask_path = ct24_tumor_mask_path[0]
	else:
		ct24_tumor_mask_path = join(mask_dir, "tumor_24h.ids")
	if len(ct24_liver_mask_path) > 0:
		ct24_liver_mask_path = ct24_liver_mask_path[0]
	else:
		ct24_liver_mask_path = join(mask_dir, "wholeliver_24hCT.ids")

	mribl_art_path = join(target_dir, patient_id, "MRI-BL", "mribl_art.nii.gz")
	mribl_pre_path = join(target_dir, patient_id, "MRI-BL", "mribl_pre.nii.gz")
	mribl_tumor_mask_path = join(mask_dir, "tumor_BL_MRI")
	mribl_liver_mask_path = join(mask_dir, "mribl_liver")
	mribl_enh_mask_path = join(mask_dir, "enh_bl")
	mribl_nec_mask_path = join(mask_dir, "nec_bl")
	#ct24_bl_enh_mask_path = join(mask_dir, "ct24_bl_enh")
	#ct24_bl_nec_mask_path = join(mask_dir, "ct24_bl_nec")

	mri30d_art_path = join(target_dir, patient_id, "MRI-30d", "mri30d_art.nii.gz")
	mri30d_pre_path = join(target_dir, patient_id, "MRI-30d", "mri30d_pre.nii.gz")
	mri30d_tumor_mask_path = join(mask_dir, "tumor_30dFU_MRI")
	mri30d_liver_mask_path = join(mask_dir, "mri30d_liver")
	mri30d_enh_mask_path = join(mask_dir, "enh_30d")
	mri30d_nec_mask_path = join(mask_dir, "nec_30d")
	#ct24_30d_enh_mask_path = join(mask_dir, "ct24_30d_enh")
	#ct24_30d_nec_mask_path = join(mask_dir, "ct24_30d_nec")

	paths = [mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
			mribl_art_path, mribl_pre_path, \
			mribl_tumor_mask_path, mribl_liver_mask_path, \
			mribl_enh_mask_path, mribl_nec_mask_path, \
			mri30d_art_path, mri30d_pre_path, \
			mri30d_tumor_mask_path, mri30d_liver_mask_path, \
			mri30d_enh_mask_path, mri30d_nec_mask_path]

	flag = False
	for path in paths:
		if not exists(path) and not exists(path+".ics"):
			print(path, "does not exist!")
			flag=True
			if check_valid:
				raise ValueError(path)

	ball_ct24_path = join(nii_dir, "ct24_ball.nii")
	ball_mribl_path = join(nii_dir, "mribl_ball.nii")
	ball_mri30d_path = join(nii_dir, "mri30d_ball.nii")
	ball_mask_path = join(mask_dir, "ball_mask")
	ball_mribl_enh_mask_path = join(mask_dir, "ball_mribl_enh_mask")
	ball_mri30d_enh_mask_path = join(mask_dir, "ball_mri30d_enh_mask")

	midlip_mask_path = join(mask_dir, "mid_lipiodol")
	ball_midlip_mask_path = join(mask_dir, "ball_mid_lipiodol")
	highlip_mask_path = join(mask_dir, "high_lipiodol")
	ball_highlip_mask_path = join(mask_dir, "ball_lipiodol")

	paths += [ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
			ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
			midlip_mask_path, ball_midlip_mask_path, \
			highlip_mask_path, ball_highlip_mask_path]

	#if flag:
	#	paths[-1] = None
	return paths


###########################
### Segmentation methods
###########################

def seg_lipiodol(patient_id, target_dir):
	paths = get_paths(patient_id, target_dir, check_valid=False)

	mask_dir, nii_dir, ct24_path, ct24_tumor_mask_path, ct24_liver_mask_path, \
	mribl_art_path, mribl_pre_path, \
	mribl_tumor_mask_path, mribl_liver_mask_path, \
	mribl_enh_mask_path, mribl_nec_mask_path, \
	mri30d_art_path, mri30d_pre_path, \
	mri30d_tumor_mask_path, mri30d_liver_mask_path, \
	mri30d_enh_mask_path, mri30d_nec_mask_path, \
	ball_ct24_path, ball_mribl_path, ball_mri30d_path, \
	ball_mask_path, ball_mribl_enh_mask_path, ball_mri30d_enh_mask_path, \
	midlip_mask_path, ball_midlip_mask_path, \
	highlip_mask_path, ball_highlip_mask_path = paths
	
	img, dims = hf.nii_load(ct24_path)

	low_mask = copy.deepcopy(img)
	low_mask = low_mask > 100
	low_mask = binary_opening(binary_closing(low_mask, structure=np.ones((2,2,1))), structure=np.ones((2,2,1)))
	mid_mask = copy.deepcopy(img)
	mid_mask = mid_mask > 150
	mid_mask = binary_closing(mid_mask)
	high_mask = copy.deepcopy(img)
	high_mask = high_mask > 200

	mask,_ = masks.get_mask(ct24_liver_mask_path, dims, img.shape)
	low_mask = low_mask*mask
	mid_mask = mid_mask*mask
	high_mask = high_mask*mask
	
	masks.save_mask(low_mask, join(mask_dir, "low_lipiodol"), dims, save_mesh=True)
	masks.save_mask(mid_mask, join(mask_dir, "mid_lipiodol"), dims, save_mesh=True)
	masks.save_mask(high_mask, join(mask_dir, "high_lipiodol"), dims, save_mesh=True)

def seg_target_lipiodol(img, save_folder, ct_dims, num_tumors=1):
	mid_mask = copy.deepcopy(img)
	mid_mask = mid_mask > 150
	mid_mask = binary_closing(mid_mask)
	
	mask1 = copy.deepcopy(img)
	mask1 = mask1 > 150
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
	x, _ = tr.rescale_img(x, C.dims)
	
	y = model.predict(np.expand_dims(x,0))[0]
	liver_mask = (y[:,:,:,1] > y[:,:,:,0]).astype(float)
	liver_mask, _ = tr.rescale_img(liver_mask, scale_shape)#orig_shape)
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
	x, _ = tr.rescale_img(x, C.dims)
	
	y = model.predict(np.expand_dims(x,0))[0]
	liver_mask = (y[:,:,:,1] > y[:,:,:,0]).astype(float)
	liver_mask[x < 30] = 0
	liver_mask, _ = tr.rescale_img(liver_mask, scale_shape)
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