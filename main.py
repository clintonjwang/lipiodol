import argparse
import config
import easygui
import niftiutils.helper_fxns as hf
import niftiutils.masks as masks
import keras
import lipiodol_methods as lm
import lipiodol_vis as lvis
import lipiodol_analysis as lan
import glob
import numpy as np
import os
from os.path import *

def get_paths_dict(out_dir):
	if not exists(out_dir):
		os.makedirs(out_dir)
	mask_dir = out_dir
	nii_dir = out_dir

	P = {'mask':mask_dir, 'nii':nii_dir, 'mrbl':{}, 'ct24':{}, 'mr30':{},
		'ball':{'mrbl':{}, 'ct24':{}, 'mr30':{}}, 'mr30Tx':{'mrbl':{}, 'ct24':{}},
			'ct24Tx':{'mrbl':{}, 'crop':{}, 'mr30':{}}}

	P['ct24']['img'] = join(nii_dir, "ct24.nii.gz")
	P['ct24']['tumor'] = join(mask_dir, "tumor")
	P['ct24']['liver'] = join(mask_dir, "liver")
	P['ct24']['lowlip'] = join(mask_dir, "lipiodol_low")
	P['ct24']['midlip'] = join(mask_dir, "lipiodol_mid")
	P['ct24']['highlip'] = join(mask_dir, "lipiodol_high")

	P['mrbl']['art'] = join(out_dir, "mrbl_art.nii.gz")
	P['mrbl']['pre'] = join(out_dir, "mrbl_pre.nii.gz")
	P['mrbl']['sub'] = join(out_dir, "mrbl_sub.nii.gz")
	P['mrbl']['equ'] = join(out_dir, "mrbl_equ.nii.gz")
	P['mr30']['art'] = join(out_dir, "mr30_art.nii.gz")
	P['mr30']['pre'] = join(out_dir, "mr30_pre.nii.gz")
	P['mr30']['sub'] = join(out_dir, "mr30_sub.nii.gz")
	P['mr30']['equ'] = join(out_dir, "mr30_equ.nii.gz")

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
	P['ct24Tx']['mrbl']['sub'] = join(nii_dir, "reg", "ct24Tx-mrbl-sub.nii")
	P['ct24Tx']['mrbl']['equ'] = join(nii_dir, "reg", "ct24Tx-mrbl-equ.nii")
	P['ct24Tx']['mrbl']['enh'] = join(mask_dir, "reg", "ct24Tx-mrbl-enh")
	P['ct24Tx']['mr30']['art'] = join(nii_dir, "reg", "ct24Tx-mr30-art.nii")
	P['ct24Tx']['mr30']['enh'] = join(mask_dir, "reg", "ct24Tx-mr30-enh")
	P['ct24Tx']['crop']['img'] = join(nii_dir, "reg", "ct24Tx-crop-img.nii")
	P['ct24Tx']['crop']['tumor'] = join(mask_dir, "reg", "ct24Tx-crop-tumor")
	P['ct24Tx']['crop']['midlip'] = join(mask_dir, "reg", "ct24Tx-crop-midlip")
	P['ct24Tx']['crop']['highlip'] = join(mask_dir, "reg", "ct24Tx-crop-highlip")

	return P

def get_mrbl(P):
	read_dir = easygui.diropenbox(msg='Select the folder containing the pre-procedural MRI (arterial phase only).')
	if read_dir is None:
		return None

	img,D = hf.dcm_load(read_dir, True, True)
	hf.save_nii(img, P['mr30']['art'], D)

	P['mr30']['tumor'] = easygui.fileopenbox(msg='Do you have the tumor mask for the MRI? If so, select the .ids file.')
	model = keras.models.load_model(join(config.Config().model_dir, "mri_liver.hdf5"))
	lm.seg_liver_mri_from_path(P['mr30']['art'], P['mr30']['liver'], model, P['mr30']['tumor'])

def get_ct24(P):
	read_dir = easygui.diropenbox(msg='Select the folder containing the post-procedural CT DICOM.')
	if read_dir is None:
		return None

	img,D = hf.dcm_load(read_dir, True, True)
	hf.save_nii(img, P['ct24']['img'], D)

	model = keras.models.load_model(join(config.Config().model_dir, "ct_liver.hdf5"))
	tmp = easygui.fileopenbox(('Do you have the tumor mask for the CT? If so, select the .ids file.',
							' Otherwise the algorithm will attempt to guess the tumor location.',
							' If the tumor is multifocal, only the largest will be kept.'))
	if tmp is not None:
		P['ct24']['tumor'] = tmp[:-4]
		masks.restrict_mask_to_largest(P['ct24']['tumor'], img_path=P['ct24']['img'])

	lm.seg_liver_ct(P['ct24']['img'], P['ct24']['liver'], model, P['ct24']['tumor'])

	if tmp is None:
		seg_tumor_ct(P)

def get_inputs_gui():
	"""UI flow. Returns None if cancelled or terminated with error,
	else returns [user, pw, acc_nums, save_dir]."""
	try:
		title = "Lipiodol Analysis Tool"
		msg = "What analysis do you want to perform?"
		choices = ["Characterize Lipiodol deposition on post-procedural CT",
				"Predict short-term Lipiodol deposition from pre-procedural MR",
				"Predict short-term tumor response from pre-procedural MR and post-procedural CT"]
		reply = easygui.choicebox(msg, title, choices)
		if reply == choices[0]:
			needed_inputs = ["ct24"]
		elif reply == choices[1]:
			needed_inputs = ["mrbl"]
		elif reply == choices[2]:
			needed_inputs = ["mrbl", "ct24"]
		else:
			return None

		out_dir = easygui.diropenbox(msg='Select a folder to save all files.')
		if out_dir is None:
			return None
		P = get_paths_dict(out_dir)

		if "ct24" in needed_inputs:
			get_ct24(P)
		if "mrbl" in needed_inputs:
			get_mrbl(P)

		if reply == choices[0]:
			lm.seg_lipiodol(P)
			img,D = hf.nii_load(P['ct24']['img'])
			M = masks.get_mask(P['ct24']['tumor'], D, img.shape)[0]
			M = M/M.max()
			V = M.sum() * np.product(D) / 1000
			L = (img > 75).sum() * np.product(D) / 1000
			tumor = img*M
			targL = (tumor > 75).sum() * np.product(D) / 1000
			lowTL = ((tumor > 75) & (tumor <= 160)).sum() * np.product(D) / 1000
			midTL = ((tumor > 160) & (tumor <= 215)).sum() * np.product(D) / 1000
			highTL = (tumor > 215).sum() * np.product(D) / 1000
			#pattern = lan.get_row_entry(P)
			easygui.msgbox(('Segmentations have been generated for Lipiodol deposition.\n\n',
							' Lipiodol coverage: %.2fcc\n' % L,
							' Target tumor volume: %.2fcc\n' % V,
							' Target Lipiodol coverage: %.2fcc (%d%% of the tumor)\n' % (targL, targL*100/V),
							'\t- Low density (75-160 HU): %.2fcc (%d%% of the tumor)\n' % (lowTL, lowTL*100/V),
							'\t- Medium density (160-215 HU): %.2fcc (%d%% of the tumor)\n' % (midTL, midTL*100/V),
							'\t- High density (>215 HU): %.2fcc (%d%% of the tumor)\n' % (highTL, highTL*100/V)))
			return None

		elif reply == choices[1]:
			raise ValueError("Not yet available")
			lm.seg_lipiodol(P)
			easygui.msgbox(('Segmentations have been generated for predicted Lipiodol deposition.\n\n',
							' Tumor volume: %.2fcc' % V,
							' Lipiodol coverage: %.2fcc' % targL,
							' Target Lipiodol coverage: %.2fcc (%d%% of the tumor)\n' % (targL, targL*100/V),
							'\t- Low density (75-160 HU): %.2fcc (%d%% of the tumor)\n' % (lowTL, lowTL*100/V),
							'\t- Medium density (160-215 HU): %.2fcc (%d%% of the tumor)\n' % (midTL, midTL*100/V),
							'\t- High density (>215 HU): %.2fcc (%d%% of the tumor)\n' % (highTL, highTL*100/V),
							'Pattern of target Lipiodol deposition: %s\n' % pattern), 'Results')
			return None

		elif reply == choices[2]:
			raise ValueError("Not yet available")

	except:
		easygui.exceptionbox()
		return None

	return [user, pw, query_terms, options]

def determine_needed_inputs(analyses):
	needed_inputs = []
	{"quant-lip-dep": ["ct24-img"],
	"quant-target-dep": ["ct24-img", "ct24-mask"],
	"pred-tumor-resp": ["mrbl-img", "mrbl-mask", "ct24-img", "ct24-mask"],
	"pred-lip-dep": ["mrbl-img"],
	}
	if "target" in analyses:
		needed_inputs.append("ct24-img")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Analyzes BL MR and 24h CT for a patient receiving cTACE.')
	#parser.add_argument('--mrbl', help='DICOM directory for baseline MR')
	#parser.add_argument('--ct24', help='DICOM directory for 24h CT')
	args = parser.parse_args()

	get_inputs_gui()

	#s = time.time()
	#img, D = hf.dcm_load(args.mrbl, True, True)
	#print("Time to convert dcm to npy: %s" % str(time.time() - s))

	#s = time.time()
	#img, D = hf.dcm_load(args.ct24, True, True)
	#lm.seg_target_lipiodol(img)
	#print("Time to load voi coordinates: %s" % str(time.time() - s))

	print("Finished!")