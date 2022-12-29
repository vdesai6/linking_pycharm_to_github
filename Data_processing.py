import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import os
from os import listdir
from sklearn.model_selection import train_test_split
import shutil


#Image and dark matter value filepaths
imgs_filepath = r"/cosma7/data/Eagle/web-storage/RefL0100N1504_Subhalo"
DM_path = r"/cosma5/data/durham/dc-desa1/project/data/Matched_Eagle_COSMA.txt"
new_folder = r"/cosma5/data/durham/dc-desa1/project/data/Galaxy_imgs"


dm = pd.read_csv(DM_path, sep=", ", header=None, names=["DM", "ID"])
dm.DM = dm.DM.astype(int)
print(dm)
print(len(dm.ID))

#galrand_imgs = Path(imgs_filepath).glob("galrand_*.png")

for e in dm.ID:
        for filename in glob.glob(os.path.join(imgs_filepath, f'galrand_{e}.png')):
                shutil.copy(filename, new_folder)


imgs = Path(new_folder)
filepath_imgs = pd.Series(list(imgs.glob("*.png")), name="file paths").astype(str)

#filepath_imgs to a dataframe
filepath_imgs = filepath_imgs.to_frame()

# Extract the ID from the file name of each image
filepath_imgs['ID'] = filepath_imgs["file paths"].apply(lambda x: x.split('_')[-1][:-4])

# Convert the ID column to integer type
filepath_imgs['ID'] = filepath_imgs['ID'].astype(int)

# Merge the filepath_imgs and dm dataframes on the ID column
images = filepath_imgs.merge(dm, left_on='ID', right_on='ID')
images = images.drop("ID", axis=1)
images = images.sample(frac=1.0, random_state=42)
print(images)

train_df, test_df = train_test_split(images, train_size=0.75, random_state=42)
t = len(train_df)
tt = len(test_df)
print(f"The length of the train data is {t} and the length of the test data is {tt}")
