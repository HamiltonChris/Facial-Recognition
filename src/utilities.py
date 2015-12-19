# data_processing.py 
# used to turn image data into an appropriate form and vice versa

import os
import numpy as np
import cv2

path = '../data/'

# flatten_image: turns a image into a 1xN vector
def flatten_image(img):
    # flatten in row major order
    rows, columns = img.shape
    return img.reshape((rows*columns,1),order = 'C')

# unflatten_image: turns a 1xN vector into an image
def unflatten_image(vector, rows, columns):
    # reshapes in row-major order 
    image = vector.reshape((rows,columns), order = 'C')
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

# turns on webcam and displays the feed to the user, when ready the user presses
# the space bar to take a picture which is then displayed to the user and after a
# keypress is formatted and turn into a flat vector.
def create_image(rows=0,columns=0,scaling=0):
    # uses default camera on computer, often the builtin webcam
    camera = cv2.VideoCapture(0)
    while(True):
        ret, img = camera.read()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape
        if rows is 0:
            rows = height
        if columns is 0:
            columns = width
        if scaling is 0:
            scaling = 1
        column_offset = np.uint32(np.floor((width - columns * scaling) / 2))
        if column_offset < 0:
            columns_offset = 0
        row_offset = np.uint32(np.floor((height - rows * scaling) / 2))
        if row_offset < 0:
            row_offset = 0
        # prints frame so that face will fit in image
        cv2.rectangle(frame,
                    (column_offset, row_offset), 
                    (column_offset + scaling * columns,row_offset + scaling * rows), 
                    (0,255,0), 
                    5)
        cv2.imshow('Press the space bar to take a picture.',frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    # consider allowing retakes
    ret, img = camera.read() 
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    camera.release()
    cv2.destroyAllWindows()
    image = format_image(image, scaling * rows, scaling * columns)
    print image.shape
    image = scale_image(image, rows, columns)
    print image.shape
    display_image(image)
    return flatten_image(image) 

    
# loads a saved image called filename and formats it to specified rows and columns
# (same size if not specified) and outputs as a flatvector
def load_image(filename, rows=0, columns=0):
    image_path = path + filename
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print image_path + " doesn't contain an image."
        return 
    img = format_image(img, rows, columns)
    return flatten_image(img)


# load_images: loads images as greyscale from data set and formats them to input 
# rows and columns outputs the formated images as a matrix of flat image vectors
def load_images(dataset, rows=0, columns=0):
    data_path = path + dataset 
    M = np.array([])
    init = False
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            img = cv2.imread(os.path.join(root, name),cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = format_image(img, rows, columns)
            vector = flatten_image(img)
            if init is False:
                M = vector
                init = True
            else:
                M = np.hstack((M, vector))
    return M

def format_image(img, rows, columns):
    height, width = img.shape
    if  not rows is 0 and rows < height:
        offset = np.floor((height - rows) / 2)
        img = img[offset:(rows + offset),:]
    if not columns is 0 and columns < width:
        offset = np.floor((width - columns) / 2)
        img = img[:,offset:(columns + offset)]
    return img

def scale_image(image, rows=0, columns=0):
    height, width = image.shape
    if rows is 0 or rows > height:
        rows = height
    if columns is 0 or columns > width:
        columns = width
    # dimensions are backwards here for whatever reason
    return cv2.resize(image, (columns,rows))

def normalize_image(vector):
    minimum = np.amin(vector)
    maximum = np.amax(vector)
    return np.uint8(((vector - minimum) / (maximum - minimum)) * 255)
