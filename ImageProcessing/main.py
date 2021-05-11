from functions import preprocessing
from functions import segmentation
from functions import colorFeature
from functions import textureFeature
from functions import rgbHistogram


all_features_data={}
# function to get all the feature data of one image
def image_processing(img,img_name,disease_type):
    pre_processed_image=preprocessing.pre_process_image(img) #preprocess the image
    segmented_image=segmentation.segment_image(pre_processed_image) #segment the image
    colorFeatureData=colorFeature.get_image_color_features(segmented_image) # extract color features
    textureFeatureData=textureFeature.get_image_texture_features(segmented_image) # extract texture features
    rgbHistogramFeatureData=rgbHistogram.get_rgbHistogram_features(segmented_image) # extract rgb histogram features

    file_details={"DISEASE TYPE":disease_type, "FILE NAME":img_name} # details about the image being considered
    for d in (file_details,colorFeatureData,textureFeatureData,rgbHistogramFeatureData):
        all_features_data.update(d) # all the data required for Machine Learning fused in a single dictionary.

    return all_features_data