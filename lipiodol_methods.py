import config
import copy
import niftiutils.masks as masks
import numpy as np
from os.path import *
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation
from skimage.morphology import ball, label

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

def seg_liver_mri(mri_img, save_folder, mri_dims, model):
    """Use a UNet to segment liver on MRI"""

    C = config.Config()
    #correct bias field!

def seg_liver_ct(ct_img, save_folder, ct_dims, model):
    """Use a UNet to segment liver on CT"""

    C = config.Config()

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
    liver_mask, _ = tr.rescale_img(liver_mask, scale_shape)#orig_shape)
    liver_mask = np.pad(liver_mask, ((crops[0], orig_shape[0]-crops[1]),
                                     (crops[2], orig_shape[1]-crops[3]),
                                     (crops[4], orig_shape[2]-crops[5])), 'constant')

    B3 = ball(3)
    B3 = B3[:,:,[0,2,3,4,6]]
    #B3 = ball(4)
    #B3 = B3[:,:,[1,3,5,6,8]]
    liver_mask = binary_opening(binary_closing(liver_mask, B3, 1), B3, 1)

    labels, num_labels = label(liver_mask, return_num=True)
    label_sizes = [np.sum(labels == label_id) for label_id in range(1,num_labels+1)]
    biggest_label = label_sizes.index(max(label_sizes))+1
    liver_mask[labels != biggest_label] = 0
    
    masks.save_mask(liver_mask, join(save_folder, "liver_pred"), ct_dims, save_mesh=True)