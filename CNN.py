import glob
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
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
dm.dropna(inplace=True)
dm.ID = dm.ID.astype(int)
dm.DM = dm.DM.astype(float)

#galrand_imgs = Path(imgs_filepath).glob("galrand_*.png")

for e in dm.ID:
        for filename in glob.glob(os.path.join(imgs_filepath, f'galrand_{e}.png')):
                shutil.copy(filename, new_folder)

imgs = Path(new_folder)
filepaths_imgs = pd.Series(list(imgs.glob("*.png")), name="file paths").astype(str)


images = pd.concat([filepaths_imgs, dm.DM], axis=1).sample(frac=1.0,
                                                           random_state=1).reset_index(drop=True)

train_df, test_df = train_test_split(images, train_size=0.7, random_state=1)

gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.25)
gen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


#generate the arrays that flow into the network using flow_from_dataframe
train_imgs = gen_train.flow_from_dataframe(dataframe=train_df,
                                           x_col="file paths",
                                           y_col="Dark Matter Mass",
                                           target_size=(256, 256),
                                           color_mode="rgb",
                                           class_mode="raw",
                                           batch_size=32,
                                           shuffle=True,
                                           seed=42,
                                           subset="training"
                                           )

val_imgs = gen_train.flow_from_dataframe(dataframe=train_df,
                                         x_col="file paths",
                                         y_col="Dark Matter Mass",
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         class_mode="raw",
                                         batch_size=32,
                                         shuffle=True,
                                         seed=42,
                                         subset="validation"
                                         )

test_imgs = gen_train.flow_from_dataframe(dataframe=test_df,
                                          x_col="file paths",
                                          y_col="Dark Matter Mass",
                                          target_size=(256, 256),
                                          color_mode="rgb",
                                          class_mode="raw",
                                          batch_size=32,
                                          shuffle=False
                                          )

#Build the network
inputs = tf.keras.Input(shape=(125, 125, 3))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="linear")(x)

epochs = 50
patience = 15

age_model = tf.keras.Model(inputs=inputs, outputs=outputs)
age_model.compile(optimizer="adam", loss="mse")
history = age_model.fit(train_imgs, validation_data=val_imgs, epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                    patience=patience,
                                                                    restore_best_weights=True)])

