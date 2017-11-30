# import required modules
from PIL import Image
import numpy as np
import math

# define color constants
class1color = [0, 0, 255]  # blue
class2color = [153, 0, 76]  # maroon
class3color = [255, 128, 0]  # orange
class4color = [255, 204, 153]  # light brown
class5color = [125, 206, 250]  # light blue
class6color = [0, 153, 0]  # green
class7color = [255, 0, 0]  # red

# read training data
index = np.genfromtxt("TRAINING WINDOW INDEXES.dat")
index = index.astype(int)

# define band combinations used
bands = ["IGB.TIF", "GREEN BAND.TIF", "BLUE BAND.TIF", "RIB.TIF", "RED BAND.TIF", "GRI.TIF"]


# calculate probability
def probability(pix_val, m, s):
    prob_val = ((1 / (s * math.sqrt((2 * math.pi)))) * np.exp(-0.5 * (((pix_val - m) / s) ** 2)))
    return prob_val


# load the image and convert it to gray scale
def loadImage(img_name):
    return Image.open(img_name).convert('L')


# calculate mean and standard deviation for each class
def getMeanAndSD(image):
    mean_vector = np.zeros((7, 1))
    sd_vector = np.zeros((7, 1))
    for m in range(0, 7):
        crop_rectangle = (index[m][0], index[m][1], index[m][2], index[m][3])
        cropped_im = image.crop(crop_rectangle)
        l = list(cropped_im.getdata())
        mean_vector[m] = np.mean(l)
        sd_vector[m] = np.std(l)
    return mean_vector, sd_vector


# load image as numpy 1024x1024 array and get class of pixels
def getClassMatrix(image, im_mean, im_sd):
    img_2D = np.asarray(image)  # 1024x1024
    img_1D = np.reshape(img_2D, (1, np.product(img_2D.shape)))  # 1x1048576
    pix_matrix = np.zeros((7, 1)) + img_1D  # 7x1048576
    prob_matrix = probability(pix_matrix, im_mean, im_sd)  # 7x1048576
    class_matrix = np.argmax(prob_matrix, 0) + 1  # 1X1048576
    class_matrix = np.reshape(class_matrix, (1024, 1024))
    return class_matrix


# compute the different class matrices
class_matrices = []
for img in bands:
    im = loadImage(img)
    mean, sd = getMeanAndSD(im)
    class_matrices.append(getClassMatrix(im, mean, sd))

# initialize to RED band classification to cover mutually exclusive pixels
image_data = np.zeros((1024, 1024, 3), dtype=np.uint8)
image_data[class_matrices[4] == 1] = class1color
image_data[class_matrices[4] == 2] = class2color
image_data[class_matrices[4] == 3] = class3color
image_data[class_matrices[4] == 4] = class4color
image_data[class_matrices[4] == 5] = class5color
image_data[class_matrices[4] == 6] = class6color
image_data[class_matrices[4] == 7] = class7color

# allot color to each pixel based on each class's segmentation accuracy
image_data[class_matrices[0] == 1] = class1color
image_data[class_matrices[1] == 2] = class2color
image_data[class_matrices[2] == 4] = class4color
image_data[class_matrices[5] == 5] = class5color
image_data[class_matrices[3] == 7] = class7color
image_data[class_matrices[4] == 6] = class6color

# save the classified image as CLASSIFIED IMAGE.TIF
classified_img = Image.fromarray(image_data, 'RGB')
classified_img.save('CLASSIFIED IMAGE.TIF')

# Display legend
legend_im = Image.open("legend.jpg")
classified_img.paste(legend_im,(905,0,1024,168))
classified_img.show()
