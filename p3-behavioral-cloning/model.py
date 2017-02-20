import csv
import cv2
import json
import matplotlib.image as mpimg
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import (ELU, Activation, Convolution2D, Cropping2D, Dense,
                          Dropout, Flatten, Lambda, MaxPooling2D)
from keras.models import Sequential, load_model
from keras.utils.visualize_util import plot

def generator(samples, batch_size=32):
    ''' feed data to keras fit_generator for training
    '''
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def create_model_commaai():
    ''' comma ai model
    '''
    cam_ch, cam_row, cam_col = 3, 160, 320
    ch, row, col = 3, 65, 320  # cropped image 
    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(cam_row, cam_col, cam_ch)))
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def create_model():
    """ Nvidia model
    """
    cam_ch, cam_row, cam_col = 3, 160, 320
    ch, row, col = 3, 65, 320  # cropped image 
    model = Sequential()
    # crop 
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(cam_row, cam_col, cam_ch)))
    # normalize
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5, border_mode='same', activation='relu',
                            subsample=(2, 2), input_shape=(ch, row, col)))
    model.add(Convolution2D(36, 5, 5, border_mode='same', activation='relu',
                            subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu',
                            subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', ))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', ))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    return model

def train():
    samples = []
    with open('processed_driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        #next(reader, None) # skip header
        for row in reader:
            samples.append([row[0],row[1]])
    print("samples", len(samples))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = create_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
            samples_per_epoch=len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples),
            nb_epoch=5)
    save_model(model)

def save_model(model):
    ''' save a model
    '''
    model.save("model.h5")

def show_model_info(model=None):
    ''' generate a diagram for the model and print the model summary
    '''
    if model == None:
        model = load_model("model.h5")
    plot(model, to_file='model.png', show_shapes=True)
    print(model.summary())

if __name__ == "__main__":
    train()
    #show_model_info()
    #model = create_model()
    #show_model_info(model)    
