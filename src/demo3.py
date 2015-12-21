import utilities as ut
import pca
import os

path = '../data/'

database = 'orl_faces'
subspace = 'orl_subspace.npz'
components = 400
rows = 112
columns = 92

if not os.path.exists(path + subspace):
   M = ut.load_images(database) 
   eigenvalues, W, mu = pca.create_subspace(M, components)
   pca.save_subspace(path + subspace, eigenvalues, W, mu)
else:
   eigenvalues, W, mu = pca.load_subspace(path + subspace)

img = ut.create_image(rows,columns, 3)

x = pca.project_image(img,W,mu)
y = pca.reverse_projection(x,W,mu)
image = ut.normalize_image(ut.unflatten_image(y,rows,columns))

original = ut.unflatten_image(img,rows,columns)

ut.display_image(image)
ut.display_image(original)
