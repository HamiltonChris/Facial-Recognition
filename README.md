# Facial Recognition using Eigenfaces
This is some facial recognition software implemented in python that uses principle component analysis to create eigenfaces which can be used to represent images of faces.
# Dependencies
This code runs on python 2.7 and requires the libraries numpy and opencv2
# Interface
## PCA
* ```create_subspace(M, k)``` takes a Matrix (M) of vectorized images and finds an input number (k) of eigenvectors with the highest eigenvalues as well as the mean image.
* ```save_subspace(filename, eigenvalues, eigenvectors, mean)``` saves the computed eigenvalues, eigenvectors and mean into a file.
* ```load_subspace(filename)``` loads from a saved file the eigenvalues, eigenvectors and mean of a space.
* ```project_image(y, W, mu)``` projects vector (y) onto a subspace (W) using the mean (mu) as given by the equation:
$$ x = (W^T(y - mu))^T $$
* ```reverse_projection(x, W, mu)``` projects vector (x) back into the original image space from the subspace (W) using mean (mu) as given in the equation:
$$ y = Wx^T + mu $$

## Utilities
* ```load_image(filename, rows=0, columns=0)``` load a single image from a given filepath in the ```data/``` directory and formats it to input rows and columns (if left default no formating) and outputs it as a flat vector.
* ```load_images(dataset, rows=0, columns=0)``` loads all images from a directory and all subdirectories in the ```data/``` directory and if specified crops them to an input number of rows and columns. Then it flattens each image into a one dimensional vector and appends it to the output matrix M.
* ```flatten_image(image)``` turns a two dimensional matrix into a one dimensional vector.
* ```unflatten_image(vector, rows, columns)``` turns a one dimesional vector back into an image of input dimensions.
* ```save_image(filename, image)``` saves image at given filename.
* ```display_image(image)``` prints image to the screen until a key is pressed.
* ```create_image(rows=0,columns=0)``` uses the default computer camera to take a picture using the spacebar. Image is then formatted and output as a flat vector.
* ```normalize_image(image)``` returns an image that is normalize with respect to the interval [0,255]. This is used to display the eigenfaces and the reprojected images.
