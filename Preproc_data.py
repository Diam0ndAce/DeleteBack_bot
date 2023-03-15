import numpy as np
from glob import iglob
import cv2
import albumentations as A
from config import data_dir, image_size

def Augementation(images, masks):
    new_images, new_masks = [], []
    
    # Augementation transforms
    hor_flip = A.HorizontalFlip(p=1)
    vert_flip = A.VerticalFlip(p=1)
    rot = A.RandomRotate90(p=1)
    
    transforms = hor_flip, vert_flip, rot
    
    for i,m in zip(images, masks):
        for t in transforms:
            transformed = t(image=i, mask=m)
            new_images.append(transformed['image'])
            new_masks.append(transformed['mask'])
            
    new_images, new_masks = np.array(new_images), np.array(new_masks)
    return new_images, new_masks

# Get images and masks from folders as numpy arrays 
def Data_to_Xy(path_img, path_mask):
    data_size = 500
    img, mask = sorted(iglob(data_dir+'Images/*.jpg'))[:data_size], sorted(iglob(data_dir+'Masks/*.png'))[:data_size]
    img, mask = list(map(Scale_image, img)), list(map(Scale_image, mask))

    img = np.array(img) / 255  
    mask = np.array(mask)

    mask -= 1  
    mask = mask[..., np.newaxis]
    
    return img, mask
    
# Reshape images and masks to input shape of our neural network
def Scale_image(filepath):
  if filepath.endswith("png"):
      mode = cv2.IMREAD_GRAYSCALE
  elif filepath.endswith("jpg"):
      mode = cv2.IMREAD_COLOR 
      
  data = cv2.resize(cv2.imread(filepath, mode), image_size)
  return data

# Preprocessing data, X:images y:masks
def Preproc_data(aug=True):
    X, y = Data_to_Xy(data_dir+'Images/', data_dir+'Masks/')
    if aug:
        X, y = Augementation(X, y)
        
    return X, y
