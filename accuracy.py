# prints confusion matrix, overall accuracy, kappa to a file

# import required modules
from PIL import Image
import numpy as np

image = "CLASSIFIED IMAGE.TIF"

# define color constants
class1color = (0, 0, 255)  # blue
class2color = (153, 0, 76)  # maroon
class3color = (255, 128, 0)  # orange
class4color = (255, 204, 153)  # light brown
class5color = (125, 206, 250)  # light blue
class6color = (0, 153, 0)  # green
class7color = (255, 0, 0)  # red

color_list = {class1color: 0, class2color: 1, class3color: 2, class4color: 3,
              class5color: 4, class6color: 5, class7color: 6}

# read training data
index = np.genfromtxt("TRAINING WINDOW INDEXES.dat")
index = index.astype(int)

# load the image in RGB mode
im = Image.open(image).convert('RGB')

# calculate confusion matrix
con = np.zeros((7, 7))
for a in range(7):
    for b in range(index[a][0] + 1, index[a][2] + 1):
        for c in range(index[a][1] + 1, index[a][3] + 1):
            (r1, g1, b1) = im.getpixel((b, c))
            con[color_list[(r1, g1, b1)], a] += 1

# calculate Overall accuracy and Kappa Coefficient
horizontal = con.sum(axis=0)
vertical = con.sum(axis=1)
pix_count = horizontal.sum(axis=0)

oa = 0  # calculate observed agreement/overall accuracy
for i in range(7):
    oa += con[i, i] / pix_count

ra = 0  # calculate random agreement
for i in range(7):
    ra += (horizontal[i] * vertical[i]) / (pix_count * pix_count)
kappa = (oa - ra) / (1 - ra)  # cohen's kappa

# write result to file
out_file = open("accuracy.txt", 'w')
print("\nConfusion matrix : \n", con, file=out_file)
print("\nCohen's kappa : ", kappa, file=out_file)
print("\nOverall accuracy : ", oa, file=out_file)
out_file.close()
