from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import os

image_path = "/hdd/visualgenome/images/"

image_names = os.listdir(image_path)

print(image_names[:100])

plt.switch_backend('agg')

#plt.imshow(image_path + image_names[0])

im = imread(image_path + image_names[0])
print(im.shape)