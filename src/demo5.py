import utilities as ut
import pca
import os

components = 400
rows = 112
columns = 92

path = '../data/'

database = 'orl_faces/'
subspace = 'orl_subspace.npz'
projections = 'orl_projections_' + str(components) + '.npz'
image_path = database + 's10/1.pgm'

if not os.path.exists(path + subspace):
   M = ut.load_images(database) 
   eigenvalues, W, mu = pca.create_subspace(M, components)
   pca.save_subspace(path + subspace, eigenvalues, W, mu)
else:
   eigenvalues, W, mu = pca.load_subspace(path + subspace)

if not os.path.exists(path + projections):
    for j in range(40):
        index = j + 1
        img = ut.load_image(database + 's' + str(index) + '/9.pgm')
        x = pca.project_image(img,W,mu)
        ut.save_projection('face ' + str(index), x, projections)
        print index

img = ut.load_image(image_path)
x = pca.project_image(img,W,mu)
y = pca.reverse_projection(x,W,mu)
image = ut.normalize_image(ut.unflatten_image(y,rows,columns))
ut.display_image(image, str(400) + ' components')

name, projection = pca.find_closest(x, path + projections)
closest_img = pca.reverse_projection(projection,W,mu)
closest_image = ut.normalize_image(ut.unflatten_image(closest_img,rows,columns))
print name
ut.display_image(closest_image, name[0])

