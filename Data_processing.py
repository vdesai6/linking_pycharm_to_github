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
print(dm)
print(len(dm.ID))

#galrand_imgs = Path(imgs_filepath).glob("galrand_*.png")

for e in dm.ID:
        for filename in glob.glob(os.path.join(imgs_filepath, f'galrand_{e}.png')):
                shutil.copy(filename, new_folder)


imgs = Path(new_folder)
filepath_imgs = pd.Series(list(imgs.glob("*.png")), name="file paths").astype(str)


images = pd.concat([filepath_imgs, dm.DM], axis=1).sample(frac=1.0,
                                                           random_state=1).reset_index(drop=True)
print(images)

train_df, test_df = train_test_split(images, train_size=0.7, random_state=1)