# Facial Recognition using Eigenfaces
This is some facial recognition software implemented in python that uses principle component analysis to create eigenfaces which can be used to represent images of faces.
# Interface
## PCA
* ```create_subspace(M, k)``` takes a Matrix (M) of vectorized images and finds an input number (k) of eigenvectors with the highest eigenvalues as well as the mean image.
* ```save_subspace(filename, eigenvalues, eigenvectors, mean)``` saves the computed eigenvalues, eigenvectors and mean into a file.
* ```load_subspace(filename)``` loads from a saved file the eigenvalues, eigenvectors and mean of a space.

## Utilities
* ```load_images(dataset, rows=0, columns=0)``` loads all images in a directory and if specified crops them to an input number of rows and columns. Then it flattens each image into a one dimensional vector and appends it to the output matrix M.
* ```flatten_image(image)``` turns a two dimensional matrix into a one dimensional vector.
* ```unflatten_image(vector, rows, columns)``` turns a one dimesional vector back into an image of input dimensions.
* ```save_image(filename, image)``` saves image at given filename.
* ```display_image(image)``` prints image to the screen until a key is pressed.
