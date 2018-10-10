
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# convert rgb to gray
def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img

image_color = cv2.imread('test-c.jpg') # read image with cv2  (n*m*3)
image_gray = grayConversion(image_color) # convert image into an n*m matrix
image_cgc = rgbConversion(image_gray)

print("color size: ", image_color.shape, "gray size: ", image_gray.shape) # print the size of the images
print(type(image_gray)) # print the type of the matrix
print(type(image_cgc)) 

cv2.imshow("Original", image_color) # show original image
cv2.imshow("GrayScale", image_gray) # show converted image

cv2.waitKey(0) # hold the image till any key is pressed
cv2.destroyAllWindows() # close all the image windows



#result.save('compressed.jpg')