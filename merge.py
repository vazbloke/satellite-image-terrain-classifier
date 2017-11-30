# import required modules
from PIL import Image

# load the band images
red = Image.open("RED BAND.TIF").convert('L')
blue = Image.open("BLUE BAND.TIF").convert('L')
green = Image.open("GREEN BAND.TIF").convert('L')
infrared = Image.open("INFRARED BAND.TIF").convert('L')

# save different band combinations as RGB images
Image.merge("RGB", (infrared, green, blue)).save("IGB.TIF")
Image.merge("RGB", (red, infrared, blue)).save("RIB.TIF")
Image.merge("RGB", (green, red, infrared)).save("GRI.TIF")
