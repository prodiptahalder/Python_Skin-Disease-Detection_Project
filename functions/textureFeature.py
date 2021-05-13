import mahotas as mt

def get_image_texture_features(img):
    print("Calculating Haralick's Texture features...") # User update and interaction

    #we use mahotas predefined libray functions to calculate the haralick's features.
    textures = mt.features.haralick(img, compute_14th_feature=True)
    mean = textures.mean(axis=0) #calculates the mean value of all the haralick's features over different axes.

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

    return dict