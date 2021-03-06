# pca.py
# Contains the functions to create a PCA subspace and represent a vector in that subspace
import numpy as np
from numpy import linalg as la

import utilities as utils

# create_subspace: finds the k principle components of a matrix M 
# returns eigenvalues, eigenvectors, mean
def create_subspace(M, k):
    [size, images] = M.shape 
    # calculate the mean 
    mean = np.dot(M,np.ones((images,1), dtype=np.int))/ images
    if (images > size):
        covariance = np.dot((M - mean), (M - mean).T)
        [eigenvectors, eigenvalues] = la.eigh(covariance)

    # this should usually be the case since the number of pixels in a picture is probably 
    # greater that the number of input pictures so instead of creating a huge Covariance
    # matrix which can be very large we instead calculate the eigenvectors of NxN matrix
    # and then use this to calculate the N eigenvectors of the DxD sized matrix
    else:
        L = np.dot((M - mean).T, (M - mean)) 
        [eigenvalues, eigenvectors] = la.eigh(L)
        eigenvectors = np.dot((M - mean), eigenvectors)
    # wow python no scoping in loops, it's kinda hard to take you serious as a language sometimes

    # to make the eigenvectors unit length or orthonormal
    for i in range(images):
        eigenvectors[:,i] = eigenvectors[:,i] / la.norm(eigenvectors[:,i])

    sorted_order = np.argsort(eigenvalues)
    sorted_order = np.flipud(sorted_order)

    eigenvalues = eigenvalues[sorted_order]
    eigenvectors = eigenvectors[:,sorted_order]

    principle_eigenvalues = eigenvalues[0:k]
    principle_eigenvectors = eigenvectors[:,0:k]

    return principle_eigenvalues, principle_eigenvectors, mean
    
# save_subspace: saves a subspace W it's eigenvalues and the mean vector to a text file
def save_subspace(filename, eigenvalues, eigenvectors, mean):
    np.savez(filename, eigenvalues = eigenvalues, eigenvectors = eigenvectors, mean = mean)

# load_subspace: returns subspace W from input file in string
def load_subspace(filename):
    data = np.load(filename)
    eigenvalues = data['eigenvalues']
    eigenvectors = data['eigenvectors']
    mean = data['mean']
    return eigenvalues, eigenvectors, mean

# project_image: projects an input image (y) onto a input subspace (W) with mean (mu)
# returns a projection onto W 
def project_image(y , W, mu):
   return np.dot(W.T,(y - mu)).T

# reverse_projection: projects the vector x back into the image space from subspace (W) with mean (mu)
# returns a flattened image vector
def reverse_projection(x, W, mu):
    return (np.dot(W,x.T) + mu)

# checks the distance between the input projection and all projections in the given file
def find_closest(projection, filename):
    names, projections = utils.load_projections(filename)
    closest_distance = -1
    closest_index = 0
    count = 0
    for proj in projections: 
        distance = la.norm(projection - proj[0])
        if closest_distance > distance or closest_distance is -1:
            closest_distance = distance
            closest_index = count
        count += 1
    return names[closest_index], projections[closest_index]

