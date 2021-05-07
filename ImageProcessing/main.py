from functions import preprocessing
from functions import segmentation
from functions import colorFeature
from functions import textureFeature
from functions import rgbHistogram

def image_processing(img,img_name,disease_type):
    pre_processed_image=preprocessing.pre_process_image(img)
    segmented_image=segmentation.segment_image(pre_processed_image)
    colorFeatureData=colorFeature.get_image_color_features(segmented_image)
    textureFeatureData=textureFeature.get_image_texture_features(segmented_image)
    rgbHistogramFeatureData=rgbHistogram.get_rgbHistogram_features(segmented_image)

    print(colorFeatureData)
    print(textureFeatureData)
    print(rgbHistogramFeatureData)

    #concat all three dictionaries with its image name and disease type

    return dict
