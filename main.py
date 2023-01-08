
import matplotlib.image as image
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import ast
import copy
import predict
import fit

def turn_into_image_matrix(mtrx_of_pixel_rgb, rows, columns):
    mtrx_of_pixel_rgb = (mtrx_of_pixel_rgb + 1) / 2
    return np.reshape(mtrx_of_pixel_rgb, (rows, columns, 3))




def main():
    i_mtrx_1 = image.imread('hamster.png')
    comb_1_2 = 16  # 4x4
    neuro_11 = 4 * 3
    e = 2500
    lr_1_2 = 0
    decision = '0'

    mode = input(":0 - fit\n1 - predict")

    if mode == '0':
        fit.fit(i_mtrx_1, neuro_11, lr_1_2)
    elif mode == '1':
        predict.predict(decision)



main()