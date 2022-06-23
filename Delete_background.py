import numpy as np
from tensorflow import keras
import cv2
from config import image_size, model_filepath


def Get_mask(img):
        model = keras.models.load_model(model_filepath)
        mask = model(cv2.resize(img, image_size)[np.newaxis, ...] / 255)[0]
        
        mask = np.argmax(mask, axis=-1)
        #mask = np.where(mask==0, 1, 0)   # delete back and argue pixels
        mask = np.where(mask%2==0, 1, 0)  #  delete only back
        
        mask_shape = (img.shape[1], img.shape[0])
        mask = cv2.resize(mask.astype(np.uint8), mask_shape)
        mask = np.array(mask)[..., np.newaxis]
        return mask
        
def Delete_back(img):
        mask = Get_mask(img)
        
        img_without_back = img * mask 
        img_without_back =  img_without_back.astype(np.uint8)
        #cv2.imwrite("test1.jpg", img_without_back)
        return img_without_back
 
        