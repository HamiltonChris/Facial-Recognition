# pca.py
# Contains the functions to create a PCA subspace and represent a vector in that subspace
import numpy as np
from numpy import linalg as la

# create_subspace: finds the k principle components of a matrix M and saves them to file
def create_subspace(M, k, filename):
    [images, size] = M.shape 
    # calculate the mean 
    mean = M * (np.ones((images,), dtype=np.int)/ images).T
    
    if (images > size):
        covariance = np.dot((M - mean).T, (M - mean))
        [eigenvectors, eigenvalues] = la.eigh(covariance)

    # this should usually be the case since the number of pixels in a picture is probably 
    # greater that the number of input pictures so instead of creating a huge Covariance
    # matrix which can be very large we instead calculate the eigenvectors of NxN matrix
    # and then use this to calculate the N eigenvectors of the DxD sized matrix
    else:
        L = np.dot((M - mean), (M - mean).T) 
        [eigenvectors, eigenvalues] = la.eigh(L)
        eigenvectors = np.dot((M - mean).T, eigenvectors)
    # wow python no scoping in loops, it's kinda hard to take you serious as a language sometimes

    sorted_order = np.argsort(eigenvalues)
    sorted_order = np.flipud(sorted_order)
    eigenvalues = eigenvalues[sorted_order]
    eigenvectors = eigenvectors[:,sorted_order]

    principle_eigenvalues = eigenvalues[0:k]
    principle_eigenvectors = eigenvectors[:,0:k]

    return principle_eigenvalues, principle_eigenvectors, mean
     

        

# project_upon: projects vector y onto subspace W


# load_subspace: returns subspace W from input file in string
