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
def display_image(image, name='image'):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# saves image in given filename 
def save_image(filename, image):
    image_path = path + filename
    cv2.imwrite(image_path, image)

# saves a given projection paired with a name
# saving additional projections must have the same dimensions
# given filename must be npz format
def save_projection(name, x, filename):
    projections = path + filename
    if not os.path.exists(projections):
        np.savez(projections, names=[name], projections=[x])
        print "created new projection file"
    else:
        data = np.load(projections)
        if data is None:
            np.savez(projections, names=[name], projections=[x])
            return 

        name_data = data['names']
        proj_data = data['projections']
        data.close() 
        print x.shape, proj_data.shape
        name_data = np.vstack((name_data, name))
        proj_data = np.vstack((proj_data, [x]))
        np.savez(projections, names=name_data, projections=proj_data)
        
# returns the first instance of name in the projection file
# filename must include filetype
def load_projection(name, filename):    
    projections = path + filename
    if not os.path.exists(projections):
        return None
    data = np.load(projections)
    names = data['names']
    count = 0
    for key in names:
        if name == key[0]:
            projection = data['projections'][count]
            data.close()
            return np.array(projection, ndmin=2).T
    data.close()
    return None

def load_projections(filename): 
    projections = path + filename
    if not os.path.exists(projections):
        return None, None
    data = np.load(projections)
    names = data['names']
    projections = data['projections']
    data.close()

    return names, projections
    

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
    image = scale_image(image, rows, columns)
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

#all Image matrix is the binary matrix of all images stitched together, all images should be 1xN, all images would be a 2d array of 1xN images
#function returns a list of coeffs, which can be used to determine which image has the largest correlation to the image
#to find which one it closest matches to, find the largest coefficient in the matrix (index 0 would be first image in allImageMatrix)
#this can also be used to generate a new image that closest resembles the target pictureMatrix using the images from allImageMatrix

#the function uses a QR decomposition to get the QR of an orthogonal matrix Q and an upper triangular matrix R
#note that a matching picture should have a coefficient of 1, non-matching pictures have really small coeffs
def compare_image_QR(targetPictureMatrix, allImageMatrix): 
    q, r = np.linalg.qr(allImageMatrix, mode='reduced');
    resultant = np.dot(np.transpose(q), targetPictureMatrix)
    coefs = np.linalg.solve(r, resultant)
    return coefs
