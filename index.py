import numpy as np
import cv2 as cv
import pandas as pd
from scipy.stats import skew
import mahotas as mt
import matplotlib.pyplot as plt
from ImageProcessing import main


image_name="skin_2.jpg"
disease_type="Melanoma"
image = cv.imread(image_name)
feature_data=main.image_processing(image,image_name,disease_type)
