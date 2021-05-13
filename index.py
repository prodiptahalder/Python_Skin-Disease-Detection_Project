import pandas as pd
import matplotlib.pyplot as plt
from ImageProcessing import get_features

#Data frame to store all the image file names of melanoma
df = pd.read_csv("image file names of the diseases in csv/mel_file_names.csv", usecols=['image_id'])

# Following loop extracts the image file name from the above dataframe and puts it into a list
file_names_list=[]
for (columnName, columnData) in df.iteritems():
    file_names_list=columnData.values

# Followiing block of code produces the required features for the given image dataset:-

i=0 # this counter serves as 1.Image-id 2.Loop control 3.Indexing the CSV file
disease_type = 1 # 1 stands for Melanoma and 0 stands for non-melanoma
for file_name in file_names_list:
    image_name = file_names_list[i]
    image_path = "melanoma_images/" + image_name
    image = plt.imread(image_path)
    print("Analysing image=" + image_name + "  Image number=" + str(i+1) + "...")# User update and interaction
    # following dictionary contains all the calculated features of an image.
    classified_features_data=get_features.image_processing(image,image_name,disease_type)
    #writing all the feature data into a csv file.
    dataset_csv_dataframe = pd.DataFrame(classified_features_data, index=[i+1])
    if i == 0:   # if the csv file is being written for the first time; mode=overwrite
        dataset_csv_dataframe.to_csv("dataset.csv")
    if i >= 1:   # if the csv file is not being written for the first time; mode=append
        dataset_csv_dataframe.to_csv("dataset.csv", mode='a', header=False)
    print("Finished Analysing image=" + image_name + "  Image number=" + str(i + 1) + ".") #User update and interaction
    i=i+1
    print()
    if i==3: break # control statement for how many pictures need to be analysed. Remove if considering all pics
