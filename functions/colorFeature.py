import numpy as np
import cv2 as cv
import colorsys

def Skew(x):
    skw = 3.0 * (np.mean(x) - np.median(x)) / np.std(x)
    return skw


def get_image_color_features(img):
    print("Calculating Color features...")
    # for RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    R, G, B = cv.split(imgRGB)

    # mean
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)
    rgb_mean_dict = dict({'mean_red': mean_R, 'mean_green': mean_G, 'mean_blue': mean_B})

    # variance
    var_R = np.var(R)
    var_G = np.var(G)
    var_B = np.var(B)
    rgb_var_dict = dict({'var_red': var_R, 'var_green': var_G, 'var_blue': var_B})

    # standard deviation
    std_R = np.std(R)
    std_G = np.std(G)
    std_B = np.std(B)
    rgb_sd_dict = dict({'sd_red': std_R, 'sd_green': std_G, 'sd_blue': std_B})

    # skewness
    sk_r = Skew(R)
    sk_g = Skew(G)
    sk_b = Skew(B)
    rgb_sk_dict = dict({'skew_red': sk_r, 'skew_green': sk_g, 'skew_blue': sk_b})

    # for HSV
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    H, S, V = cv.split(imgHSV)
    # mean
    mean_H1 = np.mean(H)
    mean_S = np.mean(S)
    mean_V = np.mean(V)
    hsv_mean_dict = dict({'mean_hue': mean_H1, 'mean_saturation': mean_S, 'mean_value': mean_V})

    # variance
    var_H1 = np.var(H)
    var_S = np.var(S)
    var_V = np.var(V)
    hsv_var_dict = dict({'var_hue': var_H1, 'var_saturation': var_S, 'var_value': var_V})

    # standard deviation
    std_H1 = np.std(H)
    std_S = np.std(S)
    std_V = np.std(V)
    hsv_sd_dict = dict({'sd_hue': std_H1, 'sd_saturation': std_S, 'sd_value': std_V})

    # skewness
    sk_h = Skew(H)
    sk_s = Skew(S)
    sk_v = Skew(V)
    hsv_sk_dict = dict({'skew_hue': sk_h, 'skew_saturation': sk_s, 'skew_value': sk_v})

    # for YCbCr
    imgYCbCr = cv.cvtColor(img, cv.COLOR_RGB2YCR_CB)

    Y, Cb, Cr = cv.split(imgYCbCr)
    # mean
    mean_Y = np.mean(Y)
    mean_Cb = np.mean(Cb)
    mean_Cr = np.mean(Cr)
    YCbCr_mean_dict = dict({'mean_Y': mean_Y, 'mean_Cb': mean_Cb, 'mean_Cr': mean_Cr})

    # variance
    var_Y = np.var(Y)
    var_Cb = np.var(Cb)
    var_Cr = np.var(Cr)
    YCbCr_var_dict = dict({'var_Y': var_Y, 'var_Cb': var_Cb, 'var_Cr': var_Cr})

    # standard deviation
    std_Y = np.std(Y)
    std_Cb = np.std(Cb)
    std_Cr = np.std(Cr)
    YCbCr_sd_dict = dict({'sd_Y': std_Y, 'sd_Cb': std_Cb, 'sd_Cr': std_Cr})

    # skewness
    sk_y = Skew(Y)
    sk_cr = Skew(Cr)
    sk_cb = Skew(Cb)
    YCbCr_sk_dict = dict({'skew_y': sk_y, 'skew_Cb': sk_cb, 'skew_Cr': sk_cr})

    # for CIEL*A*B
    imgCIELAB = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    L, A, B = cv.split(imgCIELAB)

    # mean
    mean_L = np.mean(L)
    mean_A = np.mean(A)
    mean_B = np.mean(B)
    lab_mean_dict = dict({'mean_L': mean_L, 'mean_A': mean_A, 'mean_B': mean_B})

    # variance
    var_L = np.var(L)
    var_A = np.var(A)
    var_B = np.var(B)
    lab_var_dict = dict({'var_L': var_L, 'var_A': var_A, 'var_B': var_B})

    # standard deviation
    std_L = np.std(L)
    std_A = np.std(A)
    std_B = np.std(B)
    lab_sd_dict = dict({'sd_L': std_L, 'sd_A': std_A, 'sd_B': std_B})

    # skewness
    sk_li = Skew(L)
    sk_a = Skew(A)
    sk_b = Skew(B)
    lab_sk_dict = dict({'skew_light': sk_li, 'skew_a': sk_a, 'skew_b': sk_b})

    # for CIEL*U*V
    imgCIELUV = cv.cvtColor(img, cv.COLOR_BGR2LUV)

    L, U, V = cv.split(imgCIELUV)

    # mean
    mean_L1 = np.mean(L)
    mean_U = np.mean(U)
    mean_V = np.mean(V)
    luv_mean_dict = dict({'mean_L1': mean_L1, 'mean_U': mean_U, 'mean_V': mean_V})

    # variance
    var_L1 = np.var(L)
    var_U = np.var(U)
    var_V = np.var(V)
    luv_var_dict = dict({'var_L1': var_L1, 'var_U': var_U, 'var_V': var_V})

    # standard deviation
    std_L1 = np.std(L)
    std_U = np.std(U)
    std_V = np.std(V)
    luv_sd_dict = dict({'sd_L1': std_L1, 'sd_U': std_U, 'sd_V': std_V})

    # skewness
    sk_l = Skew(L)
    sk_u = Skew(U)
    sk_v = Skew(V)
    luv_sk_dict = dict({'skew_l': sk_l, 'skew_u': sk_u, 'skew_v': sk_v})

    # bgr to NTsc

    # split Ntsc channel
    y1, i, q = colorsys.rgb_to_yiq(R, G, B)

    mean_y1 = np.mean(y1, axis=(0, 1))
    mean_i = np.mean(i, axis=(0, 1))
    mean_q = np.mean(q, axis=(0, 1))
    yiq_mean_dict = dict({'mean_y': mean_y1, 'mean_i': mean_i, 'mean_q': mean_q})

    sd_y1 = np.std(y1, axis=(0, 1))
    sd_i = np.std(i, axis=(0, 1))
    sd_q = np.std(q, axis=(0, 1))
    yiq_sd_dict = dict({'sd_y': sd_y1, 'sd_i': sd_i, 'sd_q': sd_q})

    # variance
    var_y1 = np.var(y1, axis=None)
    var_i = np.var(i, axis=None)
    var_q = np.var(q, axis=None)
    yiq_var_dict = dict({'variance_y': var_y1, 'variance_i': var_i, 'variance_q': var_q})

    # skew
    sk_y1 = Skew(y1)
    sk_i = Skew(i)
    sk_q = Skew(q)
    yiq_sk_dict = dict({'skew_y1': sk_y1, 'skew_i': sk_i, 'skew_q': sk_q})

    color_features = {}
    for d in (rgb_mean_dict, rgb_sd_dict, rgb_sk_dict, rgb_var_dict,
              hsv_mean_dict, hsv_sd_dict, hsv_sk_dict, hsv_var_dict,
              YCbCr_mean_dict, YCbCr_sd_dict, YCbCr_sk_dict, YCbCr_var_dict,
              yiq_mean_dict, yiq_sd_dict, yiq_sk_dict, yiq_var_dict,
              luv_mean_dict, luv_sd_dict, luv_sk_dict, luv_var_dict,
              lab_mean_dict, lab_sd_dict, lab_sk_dict, lab_var_dict): color_features.update(d)


    return color_features