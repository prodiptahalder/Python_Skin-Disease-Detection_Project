import cv2 as cv

def get_rgbHistogram_features(img):
    # find frequency of pixels in range 0-255
    print("Calculating RGB Histogram features...") # User update and interaction
    hist_r = cv.calcHist([img], [0], None, [16], [0, 256]) # number of pixels in the red bins
    hist_g = cv.calcHist([img], [1], None, [16], [0, 256]) # number of pixels in the green bins
    hist_b = cv.calcHist([img], [2], None, [16], [0, 256]) # number of pixels in the blue bins

    color_feature_information_dict = {}

   # following loop calculates all possible Combinations of the bin values
    for i in range(16):
        for j in range(16):
            for k in range(16):
                px_ar = (hist_r[i][0] + hist_g[j][0] + hist_b[k][0]) #calcualate the total pixel for a given combination
                px_str = str(px_ar)
                px = float(px_str)
                heading = "Red_bin-" + str(i) + " Green_bin-" + str(j) + " Blue_bin-" + str(k)
                color_feature_information_dict[heading] = px # a dictionary to store the pixel count with its combination.



    return color_feature_information_dict
