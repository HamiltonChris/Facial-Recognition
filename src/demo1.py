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

for i in range(15):
    ut.display_image(ut.unflatten_image(ut.normalize_image(W[:,i]), rows, columns), str(i))
