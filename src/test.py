# test.py
# used to quickly test functionality of code
# requires orl_faces images to be saved in data folder
import utilities as utils
import pca

components = 400
rows = 112
columns = 92

M = utils.load_images('orl_faces')
img = utils.create_image(rows, columns, 2)

eigenvalues, W, mu = pca.create_subspace(M,components)
print W.shape, img.shape, mu.shape
x = pca.project_image(img,W,mu)
print x.shape
y = pca.reverse_projection(x,W,mu)
print y.shape 
image = utils.normalize_image(utils.unflatten_image(y,rows,columns))
for i in range(1):
    print 'eigenface ' + str(i) + ': ', utils.normalize_image(utils.unflatten_image(W[:,i],rows,columns)).shape
    utils.display_image(utils.normalize_image(utils.unflatten_image(W[:,i],rows,columns)))
utils.display_image(image)
utils.display_image(utils.unflatten_image(img,rows,columns))
