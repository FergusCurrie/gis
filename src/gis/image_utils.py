import cv2 
from gis.config import Config
import numpy as np 

config = Config()



def load_image(coord):
    z,x,y = coord
    filepath = config.mnt_path / f'image/18/{z}_{x}_{y}.jpg'
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 

def load_mask(coord):
    z,x,y = coord
    filepath = config.mnt_path / f'label/18/{x}_{y}.npy'
    mask = np.load(filepath)
    return mask