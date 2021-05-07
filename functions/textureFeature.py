import numpy as np
import cv2 as cv
import pandas as pd
from scipy.stats import skew
import mahotas as mt
import matplotlib.pyplot as plt

def get_image_texture_features(img):
    textures = mt.features.haralick(img, compute_14th_feature=True)
    mean = textures.mean(axis=0)
    dict = {'Angular second moment': mean[0],
            'contrast': mean[1],
            'correlation': mean[2],
            'sum of variance': mean[3],
            'inverse difference moment': mean[4],
            'sum average': mean[5],
            'sum variance': mean[6],
            'sum entropy': mean[7],
            'entropy': mean[8],
            'difference variance': mean[9],
            'difference entropy': mean[10],
            'information measures of correlation 1': mean[11],
            'information measures of correlation 2': mean[12],
            'maximal correlation coefficient': mean[13]}

    # df = pd.DataFrame(dict,index=[0])
    # df.to_csv("textures_features_sample1.csv")

    # print(dict)
    return dict
    # dict is to be returned
