import numpy as np
import cv2 as cv
import pandas as pd
from scipy.stats import skew
import mahotas as mt
import matplotlib.pyplot as plt

def get_rgbHistogram_features(img):
    # find frequency of pixels in range 0-255
    hist_r = cv.calcHist([img], [0], None, [16], [0, 256])
    hist_g = cv.calcHist([img], [1], None, [16], [0, 256])
    hist_b = cv.calcHist([img], [2], None, [16], [0, 256])
    # show the plotting graph of an image
    # plt.plot(hist_r)
    # plt.plot(hist_g)
    # plt.plot(hist_b)
    # plt.show()
    # print(hist_r)

    pixel_count = []
    red_bin = []
    green_bin = []
    blue_bin = []

    my_dict = {}

    c = 0
    for i in range(16):
        for j in range(16):
            for k in range(16):
                px_ar = (hist_r[i][0] + hist_g[j][0] + hist_b[k][0])
                px_str = str(px_ar)
                px = float(px_str)
                # print(type(px))
                heading = "Red_bin-" + str(i) + " Green_bin-" + str(j) + " Blue_bin-" + str(k)
                my_dict[heading] = px
                pixel_count.append(px)
                red_bin.append(i)
                green_bin.append(j)
                blue_bin.append(k)

                # print(print('%d %d %d %d' % (i,j,k,(hist_r[i]+hist_g[j]+hist_b[k]))))
                # print()

    # print(pixel_count)
    # print(len(pixel_count))
    # print(my_dict)
    # dict = {'Red': red_bin, 'Green': green_bin, 'Blue': blue_bin, 'Pixel Count': pixel_count}

    # df = pd.DataFrame(my_dict,index=[0])
    # df.to_csv("Histogram_features_sample1.csv")

    # my_dict is to be returned
    return my_dict
