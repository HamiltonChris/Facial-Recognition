import utilities as utils
import pca

components = 40
rows = 112
columns = 92

M = utils.load_images('orl_faces')
img = utils.load_image('orl_faces/s10/1.pgm')

eigenvalues, W, mu = pca.create_subspace(M,components)
print W.shape, img.shape, mu.shape
x = pca.project_image(img,W,mu)
print x.shape
y = pca.reverse_projection(x,W,mu)
print y.shape 
image = utils.unflatten_image(y,rows,columns)
print image.shape
for i in range(40):
    utils.display_image(utils.unflatten_image(W[:,i],rows,columns))
utils.display_image(image)
