import utilities as ut
import pca
import os

path = '../data/'

database = 'orl_faces/'
subspace = 'orl_subspace.npz'
image_path = database + 's10/1.pgm'
components = 400
rows = 112
columns = 92

if not os.path.exists(path + subspace):
   M = ut.load_images(database) 
   eigenvalues, W, mu = pca.create_subspace(M, components)
   pca.save_subspace(path + subspace, eigenvalues, W, mu)
else:
   eigenvalues, W, mu = pca.load_subspace(path + subspace)

img = ut.load_image(image_path)
num_components = [1,2,3,4,5,6,7,8,9,10,15,20,50,75,100,150,200,400]
for i in num_components: 
    x = pca.project_image(img,W[:,1:i],mu)
    y = pca.reverse_projection(x,W[:,1:i],mu)
    image = ut.normalize_image(ut.unflatten_image(y,rows,columns))

    original = ut.unflatten_image(img,rows,columns)

    ut.display_image(image, str(i) + ' components')
    ut.display_image(original, 'original')
