# Facial Recognition using Eigenfaces
This is some facial recognition software implemented in python that uses principle component analysis to create eigenfaces which can be used to represent images of faces.
# Interface
## PCA
* create_subspace takes a Matrix of vectorized images and finds an input number of eigenvectors with the highest eigenvalues as well as the mean image.
* save_subspace saves the computed eigenvalues, eigenvectors and mean into a file
* load_subspace loads from a saved file the eigenvalues, eigenvectors and mean of a space


