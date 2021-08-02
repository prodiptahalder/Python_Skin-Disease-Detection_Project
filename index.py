import pandas as pd
import matplotlib.pyplot as plt
from ImageProcessing import main

df_mel = pd.read_csv("image file names of the diseases in csv/mel_file_names.csv", usecols=['image_id'])
df_akiec = pd.read_csv("image file names of the diseases in csv/akiec_file_names.csv", usecols=['image_id'])
df_bcc = pd.read_csv("image file names of the diseases in csv/bcc_file_names.csv", usecols=['image_id'])
df_bkl = pd.read_csv("image file names of the diseases in csv/bkl_file_names.csv", usecols=['image_id'])
df_df = pd.read_csv("image file names of the diseases in csv/df_file_names.csv", usecols=['image_id'])
df_nv = pd.read_csv("image file names of the diseases in csv/nv_file_names.csv", usecols=['image_id'])
df_vasc = pd.read_csv("image file names of the diseases in csv/vasc_file_names.csv", usecols=['image_id'])
file_names_list_mel=[]
file_names_list_akiec=[]
file_names_list_bcc=[]
file_names_list_bkl=[]
file_names_list_df=[]
file_names_list_nv=[]
file_names_list_vasc=[]

for (columnName, columnData) in df_mel.iteritems():
    file_names_list_mel=columnData.values
for (columnName, columnData) in df_akiec.iteritems():
    file_names_list_akiec=columnData.values
for (columnName, columnData) in df_bcc.iteritems():
    file_names_list_bcc=columnData.values
for (columnName, columnData) in df_bkl.iteritems():
    file_names_list_bkl=columnData.values
for (columnName, columnData) in df_df.iteritems():
    file_names_list_df=columnData.values
for (columnName, columnData) in df_nv.iteritems():
    file_names_list_nv=columnData.values
for (columnName, columnData) in df_vasc.iteritems():
    file_names_list_vasc=columnData.values

all_classified_features={}
i=0
k=1
disease_type_mel = 0
disease_type_akiec = 1
disease_type_bcc = 1
disease_type_bkl = 1
disease_type_df = 1
disease_type_nv = 1
disease_type_vasc = 1
while(1):
    #print(image_path_mel)
    #image = plt.imread(image_path_mel)
    print("Round=" + str(i + 1) + ".")
    #print("Analysing image=" + image_name_mel + "  Image number=" + str(i + 1) + ".")
    #print("Analysing image=" + image_name_akiec + "  Image number=" + str(i + 1) + ".")
    #print("Analysing image=" + image_name_bcc + "  Image number=" + str(i + 1) + ".")
    #print("Analysing image=" + image_name_bkl + "  Image number=" + str(i + 1) + ".")
    #print("Analysing image=" + image_name_df + "  Image number=" + str(i + 1) + ".")
    #print("Analysing image=" + image_name_nv + "  Image number=" + str(i + 1) + ".")
    #print("Analysing image=" + image_name_vasc + "  Image number=" + str(i + 1) + ".")

    if i <= 1111:
        image_name_mel = file_names_list_mel[i]
        image_path_mel = "assets/images/melanoma_images/" + image_name_mel
        image = plt.imread(image_path_mel)
        classified_features_data_mel=image_processing(image,image_name_mel,disease_type_mel)
        csv_dataframe_mel = pd.DataFrame(classified_features_data_mel, index=[k])
        k = k + 1
        if i == 0:
            csv_dataframe_mel.to_csv("dataset.csv")
        else:
            csv_dataframe_mel.to_csv("dataset.csv", mode='a', header=False)

    if i< 214:
        image_name_akiec = file_names_list_akiec[i]
        image_path_akiec = "assets/images/akiec/" + image_name_akiec
        image = plt.imread(image_path_akiec)
        classified_features_data_akiec = image_processing(image, image_name_mel, disease_type_akiec)
        csv_dataframe_akiec = pd.DataFrame(classified_features_data_akiec, index=[k])
        k = k + 1
        csv_dataframe_akiec.to_csv("dataset.csv", mode='a', header=False)
    if i< 229:
        image_name_bcc = file_names_list_bcc[i]
        image_path_bcc = "assets/images/bcc/" + image_name_bcc
        image = plt.imread(image_path_bcc)
        classified_features_data_bcc = image_processing(image, image_name_mel, disease_type_bcc)
        csv_dataframe_bcc = pd.DataFrame(classified_features_data_bcc, index=[k])
        k = k + 1
        csv_dataframe_bcc.to_csv("dataset.csv", mode='a', header=False)

    if i< 214:
        image_name_bkl = file_names_list_bkl[i]
        image_path_bkl = "assets/images/bkl/" + image_name_bkl
        image = plt.imread(image_path_bkl)
        classified_features_data_bkl = image_processing(image, image_name_mel, disease_type_bkl)
        csv_dataframe_bkl = pd.DataFrame(classified_features_data_bkl, index=[k])
        k = k + 1
        csv_dataframe_bkl.to_csv("dataset.csv", mode='a', header=False)

    if i< 100:
        image_name_df = file_names_list_df[i]
        image_path_df = "assets/images/df/" + image_name_df
        image = plt.imread(image_path_df)
        classified_features_data_df = image_processing(image, image_name_mel, disease_type_df)
        csv_dataframe_df = pd.DataFrame(classified_features_data_df, index=[k])
        k = k + 1
        csv_dataframe_df.to_csv("dataset.csv", mode='a', header=False)

    if i<254:
        image_name_nv = file_names_list_nv[i]
        image_path_nv = "assets/images/nv/" + image_name_nv
        image = plt.imread(image_path_nv)
        classified_features_data_nv = image_processing(image, image_name_mel, disease_type_nv)
        csv_dataframe_nv = pd.DataFrame(classified_features_data_nv, index=[k])
        k = k + 1
        csv_dataframe_nv.to_csv("dataset.csv", mode='a', header=False)

    if i< 100:
        image_name_vasc = file_names_list_vasc[i]
        image_path_vasc = "assets/images/vasc/" + image_name_vasc
        image = plt.imread(image_path_vasc)
        classified_features_data_vasc = image_processing(image, image_name_mel, disease_type_vasc)
        csv_dataframe_vasc = pd.DataFrame(classified_features_data_vasc, index=[k])
        k = k + 1
        csv_dataframe_vasc.to_csv("dataset.csv", mode='a', header=False)
    #print(classified_features_data_df)

    '''
    if i==0:
        csv_dataframe_mel.to_csv("dataset.csv")
        csv_dataframe_akiec.to_csv("dataset.csv", mode='a', header=False)
     #  csv_dataframe_bcc.to_csv("dataset.csv", mode='a', header=False)
      # csv_dataframe_bkl.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_df.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_nv.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_vasc.to_csv("dataset.csv", mode='a', header=False)

    if i>=1:
        csv_dataframe_mel.to_csv("dataset.csv", mode='a', header=False)
        csv_dataframe_akiec.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_bcc.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_bkl.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_df.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_nv.to_csv("dataset.csv", mode='a', header=False)
        #csv_dataframe_vasc.to_csv("dataset.csv", mode='a', header=False)
    '''

    #print("Finished Analysing image=" + image_name_mel + "  Image number=" + str(i + 1) + ".")
    #print("Finished Analysing image=" + image_name_akiec + "  Image number=" + str(i + 1) + ".")
    #print("Finished Analysing image=" + image_name_bcc + "  Image number=" + str(i + 1) + ".")
    #print("Finished Analysing image=" + image_name_bkl + "  Image number=" + str(i + 1) + ".")
    #print("Finished Analysing image=" + image_name_df + "  Image number=" + str(i + 1) + ".")
    #print("Finished Analysing image=" + image_name_nv + "  Image number=" + str(i + 1) + ".")
    #print("Finished Analysing image=" + image_name_vasc + "  Image number=" + str(i + 1) + ".")
    i=i+1

    #print()
    if i==1111: break # control statement

