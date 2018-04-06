import config
import copy
import niftiutils.masks as masks
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import numpy as np
import math
from math import pi, radians, degrees
from os.path import *
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation
from skimage.morphology import ball, label

###########################
### Spherical shell analysis
###########################

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

def get_avg_core_intensity(reg_img_path, ball_mask_path, r_frac=.5):
    img, dims = hf.nii_load(reg_img_path)
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
    I = np.nansum(img)
    return I/V


###########################
### Segmentation methods
###########################

def seg_lipiodol(img, save_folder, ct_dims):
	low_mask = copy.deepcopy(img)
	low_mask = low_mask > 100
	low_mask = binary_opening(binary_closing(low_mask, structure=np.ones((2,2,1))), structure=np.ones((2,2,1)))
	mid_mask = copy.deepcopy(img)
	mid_mask = mid_mask > 150
	mid_mask = binary_closing(mid_mask)
	high_mask = copy.deepcopy(img)
	high_mask = high_mask > 200
	
	masks.save_mask(low_mask, join(save_folder, "low_lipiodol"), ct_dims, save_mesh=True)
	masks.save_mask(mid_mask, join(save_folder, "mid_lipiodol"), ct_dims, save_mesh=True)
	masks.save_mask(high_mask, join(save_folder, "high_lipiodol"), ct_dims, save_mesh=True)

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
		liver_mask[tumor_mask > 128] = liver_mask.max()

	masks.save_mask(liver_mask, save_path, mri_dims, save_mesh=True)

def seg_liver_ct(ct_img, save_path, ct_dims, model):
	"""Use a UNet to segment liver on CT"""

	C = config.Config()

	orig_shape = ct_img.shape

	x = tr.apply_window(ct_img)
	x -= np.amin(x)
	x /= np.amax(x)

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
	
	masks.save_mask(liver_mask, save_path, ct_dims, save_mesh=True)