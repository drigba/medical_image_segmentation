import os
import cv2
import nibabel as nib
import numpy as np
import random
import torchvision.transforms.functional as TF


class DualTransform(object):
    def __init__(self, degree, max_translate_x, max_translate_y) -> None:
        self.degree = degree
        self.max_translate_x = max_translate_x
        self.max_translate_y = max_translate_y
    
    def __call__(self, image, segmentation):
        if random.random() > 0.5:
            image = TF.hflip(image)
            segmentation = TF.hflip(segmentation)
        if random.random() > 0.5:
            image = TF.vflip(image)
            segmentation = TF.vflip(segmentation)

        degree = random.randint(-self.degree,self.degree)
        translate_x = random.uniform(0, self.max_translate_x)
        translate_y = random.uniform(0, self.max_translate_y)
        image = TF.affine(image, degree, (translate_x, translate_y),1,0)
        segmentation = TF.affine(segmentation, degree, (translate_x, translate_y),1,0)
        return image,segmentation


def get_acdc(path,input_size=(224,224,1)):
    """
    Read images and masks for the ACDC dataset
    """
    all_imgs = []
    all_gt = []
    all_header = []
    all_affine = []
    info = []
    for root, directories, files in os.walk(path):
        for file in files:
            if ".gz" and "frame" in file:
                if "_gt" not in file:
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    all_header.append(nib.load(img_path).header)
                    all_affine.append(nib.load(img_path).affine)  
                    for idx in range(img.shape[2]):
                        i = cv2.resize(img[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
                        all_imgs.append(i)
                        
                else:
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    for idx in range(img.shape[2]):
                        i = cv2.resize(img[:,:,idx], (input_size[0], input_size[1]), interpolation=cv2.INTER_NEAREST)
                        all_gt.append(i)
            elif '.cfg' in file:
                with open(root + "/" + file, 'r') as file:
                    data = {}
                    for line in file:
                        label, value = line.strip().split(": ")
                        data[label] = value
                    info.append(data)

    data = [all_imgs, all_gt, info]                  
 
    data[0] = np.expand_dims(data[0], axis=3)
    if path[-9:] != "true_test":
        data[1] = np.expand_dims(data[1], axis=3)
    
    return data, all_affine, all_header

def convert_masks(y, data="acdc"):
    """
    Given one masks with many classes create one mask per class
    """

    masks = np.zeros((y.shape[0], y.shape[1], y.shape[2], 4))
    
    for i in range(y.shape[0]):
        masks[i][:,:,0] = np.where(y[i]==0, 1, 0)[:,:,-1] 
        masks[i][:,:,1] = np.where(y[i]==1, 1, 0)[:,:,-1] 
        masks[i][:,:,2] = np.where(y[i]==2, 1, 0)[:,:,-1] 
        masks[i][:,:,3] = np.where(y[i]==3, 1, 0)[:,:,-1]
            
    return masks

def convert_mask_single(y):
    """
    Given one masks with many classes create one mask per class
    y: shape (w,h)
    """
    mask = np.zeros((4, y.shape[0], y.shape[1]))
    mask[0, :, :] = np.where(y == 0, 1, 0)
    mask[1, :, :] = np.where(y == 1, 1, 0)
    mask[2, :, :] = np.where(y == 2, 1, 0)
    mask[3, :, :] = np.where(y == 3, 1, 0)

    return mask

def visualize(image_raw,mask):
    """
    iamge_raw:gray image with shape [width,height,1]
    mask: segment mask image with shape [num_class,width,height]
    this function return an image using multi color to visualize masks in raw image
    """
    # Convert grayscale image to RGB
    image = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
    
    # Get the number of classes (i.e. channels) in the mask
    num_class = mask.shape[0]
    
    # Define colors for each class (using a simple color map)
    colors = []
    for i in range(1, num_class):  # skip first class (background)
        hue = int(i/float(num_class-1) * 179)
        color = np.zeros((1, 1, 3), dtype=np.uint8)
        color[0, 0, 0] = hue
        color[0, 0, 1:] = 255
        color = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
        colors.append(color)

    # Overlay each non-background class mask with a different color on the original image
    for i in range(1, num_class):
        class_mask = mask[i, :, :]
        class_mask = np.repeat(class_mask[:, :, np.newaxis], 3, axis=2)
        class_mask = class_mask.astype(image.dtype)
        class_mask = class_mask * colors[i-1]
        image = cv2.addWeighted(image, 1.0, class_mask, 0.5, 0.0)

    return image

def get_images_with_info(path):
    all_imgs = []
    all_gts = []
    infos = []
    for root, directories, files in os.walk(path):
        current_group = ""
        current_ES = ""
        for file in files:
            if '.cfg' in file:
                with open(root + "/" + file, 'r') as file:
                    data = {}
                    for line in file:
                        label, value = line.strip().split(": ")
                        data[label] = value
                    current_group = data['Group']
                    current_ES = data['ES']
                    infos.append(data)
            elif ".gz" and "frame" in file:
                if "_gt" not in file:
                    phase = ""
                    if int(current_ES) == int(file[16:18]):
                        phase = "ES"
                    else:
                        phase = "ED"
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    for idx in range(img.shape[2]):
                        all_imgs.append([img[:,:,idx],current_group,phase])
                else:
                    phase = ""
                    if int(current_ES) == int(file[16:18]):
                        phase = "ES"
                    else:
                        phase = "ED"
                    img_path = root + "/" + file
                    img = nib.load(img_path).get_fdata()
                    for idx in range(img.shape[2]):
                        all_gts.append([img[:,:,idx],current_group,phase])

    data = [all_imgs, all_gts, infos]  
    
    return data

def get_label_percentages(label_img):
    size = label_img.shape
    pixel_count = size[0]*size[1]
    percs = []
    for i in range(4):
        percs += [np.count_nonzero((label_img[:,:] == i))/pixel_count]
    return percs
    