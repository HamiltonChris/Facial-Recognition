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

# displays input file image
def display_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# saves image in given filename 
def save_image(filename, image):
    image_path = path + filename
    cv2.imwrite(image_path, image)

# loads a saved image called filename and formats it to specified rows and columns
# (same size if not specified) and outputs as a flatvector
def load_image(filename, rows=0, columns=0):
    image_path = path + filename
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print image_path + " doesn't contain an image."
        return 
    img = format_image(img, rows, columns)
    return flatten_image(img).T

# turns on webcam and displays the feed to the user, when ready the user presses
# the space bar to take a picture which is then displayed to the user and after a
# keypress is formatted and turn into a flat vector.
def create_image(rows=0,columns=0):
    # uses default camera on computer, often the builtin webcam
    camera = cv2.VideoCapture(0)
    while(True):
        ret, img = camera.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    camera.release()
    cv2.destroyAllWindows()
    image = format_image(image, rows, columns)
    display_image(image)
    return flatten_image(image).T 

# load_images: loads images as greyscale from data set and formats them to input 
# rows and columns outputs the formated images as a matrix of flat image vectors
def load_images(dataset, rows=0, columns=0):
    data_path = path + dataset 
    X = [] 
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            img = cv2.imread(os.path.join(root, name),cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = format_image(img, rows, columns)
            X.append(flatten_image(img).T) 
    M = np.array(X).T
    return M

def format_image(img, rows, columns):
    height, width = img.shape
    if  not rows is 0 and rows < height:
        top = np.floor((height - rows) / 2)
        img = img[top:(rows + top),:]
    if not columns is 0 and columns < width:
        left = np.floor((width - columns) / 2)
        img = img[:,left:(columns + left)]
    return img
