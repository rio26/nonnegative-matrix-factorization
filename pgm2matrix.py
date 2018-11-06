'''
Created on Oct 10th, 2018
@author: Rio Li

I know these codes look extremely painful but anyway I tried my best.
'''

import re
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab

from skimage import io
from skimage import io, transform


def read_pgm(filename, byteorder='>'):
    """
    Return image data from a raw PGM file as numpy array.
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def pgm2vector(filename):
    """Converting an image to a row vector"""
    image = io.imread(filename)
    # [row,col] = np.shape(image) # 112 * 92
    vec = np.concatenate(image) # (1 * 10304)
    # print('pgm to vector size: ',np.shape(vec))
    return vec

def pgm2matrix(filename, numberofimages):
    """Converting all the images in the file to a matrix"""
    mat = np.matrix(pgm2vector(os.path.join(filename + '1.pgm'))) # first column of the matrix
    for i in range (2, numberofimages+1): 
        col_vec = pgm2vector(os.path.join(filename + str(i) + '.pgm')) # compute the pgm_vector for the second img
        mat = np.insert(mat,i-1, col_vec, axis = 0) # add a column to the matrix
    # print(np.shape(mat.T))
    return(mat.T) # return (10 * 10304)

def matrix2png(matrix, row, col, file_store_path = None):
    [m0,m1] = np.shape(matrix)
    print('row: ', m0, '; col:', m1)
    for col_num in range(1):
        tmp = matrix[:,col_num]
        # print('tmp:' , np.shape(tmp), tmp) # tmp: (10304, 1)
        new_image = tmp.T.reshape((row, col))
        print(np.shape(new_image))
        return new_image
        

if __name__ == "__main__":
    
    # image = read_pgm(os.path.join('att_faces/s1/1.pgm'), byteorder='<')
    # plt.imshow(image, pyplot.cm.gray)
    # plt.show()
    matrix = pgm2matrix('orl_face/s1/', 10)
    image_t = matrix2png(matrix, 112, 92)
    print(type(image_t))
    imgplog = plt.imshow(image_t, cmap="gray")
    pylab.show()
    input()