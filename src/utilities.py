# data_processing.py 
# used to turn image data into an appropriate form and vice versa

import os
import numpy as np
import cv2

path = '../data/'

# flatten_image: turns a image into a 1xN vector
def flatten_image(img):
    # flatten in row major order
    return img.flatten(order = 'C')

# unflatten_image: turns a 1xN vector into an image
def unflatten_image(vector, rows, columns):
    # must traverse in row-major order 
    img = []
    for i in range(rows):
        row = vector[i*columns:(i + 1)*columns]
        img.append(row)
    image = np.array(img)
    return image

# load_images: loads images from data set and formats them to input rows and columns
def load_images(dataset, rows=0, columns=0):
    data_path = path + dataset 
    X = [] 
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            img = cv2.imread(os.path.join(root, name),0)
            if img is None:
                continue
            height, width = img.shape
            if  not rows is 0 and rows < height:
                top = np.floor((height - rows) / 2)
                img = img[top:(rows + top),:]
            if not columns is 0 and columns < width:
                left = np.floor((width - columns) / 2)
                img = img[:,left:(columns + left)]
            X.append(flatten_image(img).T) 
    M = np.array(X).T
    return M
