import numpy as np
import cv2 as cv
import pandas as pd
from scipy.stats import skew
import mahotas as mt
import matplotlib.pyplot as plt

def get_image_color_features(img):
    # for RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # cv.imshow("RGB Image",imgRGB)
    R, G, B = cv.split(imgRGB)
    # cv.imshow('Red',R)
    # cv.imshow('Green',G)
    # cv.imshow('Blue',B)

    # mean
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)
    mean_dict = dict({'mean_red': mean_R, 'mean_green': mean_G, 'mean_blue': mean_B})
    # print(mean_dict)

    # variance
    var_R = np.var(R)
    var_G = np.var(G)
    var_B = np.var(B)
    var_dict = dict({'var_red': var_R, 'var_green': var_G, 'var_blue': var_B})
    # print(var_dict)

    # standard deviation
    std_R = np.std(R)
    std_G = np.std(G)
    std_B = np.std(B)
    sd_dict = dict({'sd_red': std_R, 'sd_green': std_G, 'sd_blue': std_B})
    # print(sd_dict)

    # skewness
    sk_R = skew(R)
    sk_G = skew(G)
    sk_B = skew(B)
    sk_dict_RGB = dict({'skew_red': sk_R, 'skew_green': sk_G, 'skew_blue': sk_B})
    # print(sk_dict)

    dataFrame = pd.DataFrame(data=sk_dict_RGB)
    skewValue = dataFrame.skew();

    print(skewValue)

    # for HSV
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # cv.imshow("HSV Image",imgHSV)
    H, S, V = cv.split(imgHSV)
    # cv.imshow('Hue',H)
    # cv.imshow('Saturation',S)
    # cv.imshow('Value',V)

    # mean
    mean_H1 = np.mean(H)
    mean_S = np.mean(S)
    mean_V = np.mean(V)
    mean_dict = dict({'mean_hue': mean_H1, 'mean_saturation': mean_S, 'mean_value': mean_V})
    # print(mean_dict)

    # variance
    var_H1 = np.var(H)
    var_S = np.var(S)
    var_V = np.var(V)
    var_dict = dict({'var_hue': var_H1, 'var_saturation': var_S, 'var_value': var_V})
    # print(var_dict)

    # standard deviation
    std_H1 = np.std(H)
    std_S = np.std(S)
    std_V = np.std(V)
    sd_dict = dict({'sd_hue': std_H1, 'sd_saturation': std_S, 'sd_value': std_V})
    # print(sd_dict)

    # skewness
    sk_H1 = skew(H)
    sk_S = skew(S)
    sk_V = skew(V)
    sk_dict_HSV = dict({'skew_hue': sk_H1, 'skew_saturation': sk_S, 'skew_value': sk_V})
    # print(sk_dict)

    dataFrame = pd.DataFrame(data=sk_dict_HSV)
    skewValue = dataFrame.skew();

    print(skewValue)

    # for YCbCr
    imgYCbCr = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)
    # cv.imshow("YCbCr Image",imgYCbCr)
    Y, Cb, Cr = cv.split(imgYCbCr)
    # cv.imshow('img_Y',Y)
    # cv.imshow('img_Cb',Cb)
    # cv.imshow('img_Cr',Cr)

    # mean
    mean_Y = np.mean(Y)
    mean_Cb = np.mean(Cb)
    mean_Cr = np.mean(Cr)
    mean_dict = dict({'mean_Y': mean_Y, 'mean_Cb': mean_Cb, 'mean_Cr': mean_Cr})
    # print(mean_dict)

    # variance
    var_Y = np.var(Y)
    var_Cb = np.var(Cb)
    var_Cr = np.var(Cr)
    var_dict = dict({'var_Y': var_Y, 'var_Cb': var_Cb, 'var_Cr': var_Cr})
    # print(var_dict)

    # standard deviation
    std_Y = np.std(Y)
    std_Cb = np.std(Cb)
    std_Cr = np.std(Cr)
    sd_dict = dict({'sd_Y': std_Y, 'sd_Cb': std_Cb, 'sd_Cr': std_Cr})
    # print(sd_dict)

    # skewness
    sk_Y = skew(Y)
    sk_Cb = skew(Cb)
    sk_Cr = skew(Cr)
    sk_dict_YCbCr = dict({'skew_Y': sk_Y, 'skew_Cb': sk_Cb, 'skew_Cr': sk_Cr})
    # print(sk_dict)

    dataFrame = pd.DataFrame(data=sk_dict_YCbCr)
    skewValue = dataFrame.skew();

    print(skewValue)

    # for CIEL*A*B
    imgCIELAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # cv.imshow("LAB Image",imgCIELAB)
    L, A, B = cv.split(imgCIELAB)
    # cv.imshow('img_L',L)
    # cv.imshow('img_A',A)
    # cv.imshow('img_B',B)

    # mean
    mean_L = np.mean(L)
    mean_A = np.mean(A)
    mean_B = np.mean(B)
    mean_dict = dict({'mean_L': mean_L, 'mean_A': mean_A, 'mean_B': mean_B})
    # print(mean_dict)

    # variance
    var_L = np.var(L)
    var_A = np.var(A)
    var_B = np.var(B)
    var_dict = dict({'var_L': var_L, 'var_A': var_A, 'var_B': var_B})
    # print(var_dict)

    # standard deviation
    std_L = np.std(L)
    std_A = np.std(A)
    std_B = np.std(B)
    sd_dict = dict({'sd_L': std_L, 'sd_A': std_A, 'sd_B': std_B})
    # print(sd_dict)

    # skewness
    sk_L = skew(L)
    sk_A = skew(A)
    sk_B = skew(B)
    sk_dict_LAB = dict({'skew_L': sk_L, 'skew_A': sk_A, 'skew_B': sk_B})
    # print(sk_dict)

    dataFrame = pd.DataFrame(data=sk_dict_LAB)
    skewValue = dataFrame.skew();

    print(skewValue)

    # for CIEL*U*V
    imgCIELUV = cv.cvtColor(img, cv.COLOR_BGR2LUV)
    # cv.imshow("LUV Image",imgCIELUV)
    L, U, V = cv.split(imgCIELUV)
    # cv.imshow('img_L1',L)
    # cv.imshow('img_U',U)
    # cv.imshow('img_V',V)

    # mean
    mean_L1 = np.mean(L)
    mean_U = np.mean(U)
    mean_V = np.mean(V)
    mean_dict = dict({'mean_L1': mean_L1, 'mean_U': mean_U, 'mean_V': mean_V})
    # print(mean_dict)

    # variance
    var_L1 = np.var(L)
    var_U = np.var(U)
    var_V = np.var(V)
    var_dict = dict({'var_L1': var_L1, 'var_U': var_U, 'var_V': var_V})
    # print(var_dict)

    # standard deviation
    std_L1 = np.std(L)
    std_U = np.std(U)
    std_V = np.std(V)
    sd_dict = dict({'sd_L1': std_L1, 'sd_U': std_U, 'sd_V': std_V})
    # print(sd_dict)

    # skewness
    sk_L1 = skew(L)
    sk_U = skew(U)
    sk_V = skew(V)
    sk_dict_LUV = dict({'skew_L1': sk_L1, 'skew_U': sk_U, 'skew_V': sk_V})
    # print(sk_dict)

    dataFrame = pd.DataFrame(data=sk_dict_LUV)
    skewValue = dataFrame.skew();

    print(skewValue)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    mean = dict(
        {'mean_red': mean_R, 'mean_green': mean_G, 'mean_blue': mean_B, 'mean_hue': mean_H1, 'mean_saturation': mean_S,
         'mean_value': mean_V, 'mean_Y': mean_Y, 'mean_Cb': mean_Cb, 'mean_Cr': mean_Cr, 'mean_L': mean_L,
         'mean_A': mean_A, 'mean_B': mean_B, 'mean_L1': mean_L1, 'mean_U': mean_U, 'mean_V': mean_V})
    st_dv = dict({'sd_red': std_R, 'sd_green': std_G, 'sd_blue': std_B, 'sd_hue': std_H1, 'sd_saturation': std_S,
                  'sd_value': std_V, 'sd_Y': std_Y, 'sd_Cb': std_Cb, 'sd_Cr': std_Cr, 'sd_L': std_L, 'sd_A': std_A,
                  'sd_B': std_B, 'sd_L1': std_L1, 'sd_U': std_U, 'sd_V': std_V})
    var = dict({'var_red': var_R, 'var_green': var_G, 'var_blue': var_B, 'var_hue': var_H1, 'var_saturation': var_S,
                'var_value': var_V, 'var_Y': var_Y, 'var_Cb': var_Cb, 'var_Cr': var_Cr, 'var_L': var_L, 'var_A': var_A,
                'var_B': var_B, 'var_L1': var_L1, 'var_U': var_U, 'var_V': var_V})
    color_features = dict(
        {'mean_red': mean_R, 'mean_green': mean_G, 'mean_blue': mean_B, 'mean_hue': mean_H1, 'mean_saturation': mean_S,
         'mean_value': mean_V, 'mean_Y': mean_Y, 'mean_Cb': mean_Cb, 'mean_Cr': mean_Cr, 'mean_L': mean_L,
         'mean_A': mean_A, 'mean_B': mean_B, 'mean_L1': mean_L1, 'mean_U': mean_U, 'mean_V': mean_V,
         'sd_red': std_R, 'sd_green': std_G, 'sd_blue': std_B, 'sd_hue': std_H1, 'sd_saturation': std_S,
         'sd_value': std_V, 'sd_Y': std_Y, 'sd_Cb': std_Cb, 'sd_Cr': std_Cr, 'sd_L': std_L, 'sd_A': std_A,
         'sd_B': std_B, 'sd_L1': std_L1, 'sd_U': std_U, 'sd_V': std_V,
         'var_red': var_R, 'var_green': var_G, 'var_blue': var_B, 'var_hue': var_H1, 'var_saturation': var_S,
         'var_value': var_V, 'var_Y': var_Y, 'var_Cb': var_Cb, 'var_Cr': var_Cr, 'var_L': var_L, 'var_A': var_A,
         'var_B': var_B, 'var_L1': var_L1, 'var_U': var_U, 'var_V': var_V})

    # Return all the color feature dictionaries
    return color_features
    #def get_image_mean(img):

    #def get_image_standard_deviation(img):

    #def get_image_skewness(img):