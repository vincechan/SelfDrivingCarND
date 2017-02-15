import csv
import json
import os

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import cv2
from keras.layers import (Activation, Convolution2D, Cropping2D, Dense,
                          Dropout, Flatten, Lambda, MaxPooling2D, ELU)
from keras.models import Sequential, load_model


DATA_FOLDER = "./data/"

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
                image = cv2.imread(batch_sample[1])
                angle = float(batch_sample[0])
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def create_model_m():
    #ch, row, col = 3, 160, 320  # camera format
    ch, row, col = 3, 80, 320  # Trimmed image format

    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))            
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
    """Create model
    """
    ch, row, col = 3, 80, 320  # Trimmed image format
    model = Sequential()
    # crop image. new shape: (80, 320, 3)
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    # normalize image
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
    model.add(Dropout(0.2))
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    return model

def preprocess():
    samples = []
    with open(DATA_FOLDER + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        # loop through all samples and flip images for any sample
        # that has non-zero steering angle
        for idx, row in enumerate(reader):
            print("processing", row)

            samples.append(row)
            if idx == 0: continue #skip header
            
            steering = float(row[3])
            if steering == 0: continue #skip zero steering
            
            row_flip = row[:]
            row_flip[3] = -steering # flip angle
            dir = DATA_FOLDER + "IMG/"
            row_flip[0] = row[0].replace(".jpg", "_flip.jpg")
            row_flip[1] = row[1].replace(".jpg", "_flip.jpg")
            row_flip[2] = row[2].replace(".jpg", "_flip.jpg")
            center_flip = np.fliplr(cv2.imread(dir + row[0].split("/")[-1]))
            left_flip = np.fliplr(cv2.imread(dir + row[1].split("/")[-1]))
            right_flip = np.fliplr(cv2.imread(dir + row[2].split("/")[-1]))
            cv2.imwrite(dir + row_flip[0].split("/")[-1], center_flip)
            cv2.imwrite(dir + row_flip[1].split("/")[-1], left_flip)
            cv2.imwrite(dir + row_flip[2].split("/")[-1], right_flip)
            samples.append(row_flip)

    with open(DATA_FOLDER + 'driving_log_preprocessed.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for sample in samples:
            writer.writerow(sample)

def train():
    CORRECTION = 0.12
    samples = []
    left_samples = []
    right_samples = []
    center_samples = []
    with open(DATA_FOLDER + 'driving_log_preprocessed.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header
        for row in reader:
            steering_center = float(row[3])
            steering_left = steering_center + CORRECTION
            steering_right = steering_center - CORRECTION
            img_center = DATA_FOLDER + row[0].lstrip()
            img_left = DATA_FOLDER + row[1].lstrip()
            img_right = DATA_FOLDER + row[2].lstrip()
            if steering_center > 0.2:
                right_samples.append([steering_center, img_center])
                right_samples.append([steering_left, img_left])
                right_samples.append([steering_right, img_right])
            elif steering_center < -0.2:
                left_samples.append([steering_center, img_center])
                left_samples.append([steering_left, img_left])
                left_samples.append([steering_right, img_right])
            #elif steering_center == 0:
            else:
                center_samples.append([steering_center, img_center])
                center_samples.append([steering_left, img_left])
                center_samples.append([steering_right, img_right])
            #else:
            #    pass #ignore these samples

        shuffle(center_samples)
        samples = samples + center_samples[0:len(left_samples) * 4]
        samples = samples + left_samples
        samples = samples + right_samples
        print("left samples:", len(left_samples))
        print("right samples:", len(right_samples))
        print("straight samples:", len(center_samples))
        print("samples:", len(samples))

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

def run_test():
    ''' verify the validity of the model by checking that it will overfit when
        training data only has a few samples'''
    samples = []
    with open(DATA_FOLDER + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        next(reader, None)
        for line in reader:
            samples.append(line)
    train_generator = generator(samples, batch_size=3)
    model = create_model()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,
            samples_per_epoch=len(samples), 
            nb_epoch=100)
    save_model(model)

def save_model(model):
    ''' save a model to disc
    '''
    model.save("model.h5")

def show_model_info(model=None):
    ''' generate a diagram for the model and print the model summary
    '''
    from keras.models import load_model
    from keras.utils.visualize_util import plot
    if model == None:
        model = load_model("model.h5")
    plot(model, to_file='model.png', show_shapes=True)
    print(model.summary())

if __name__ == "__main__":
    #preprocess()    
    train()
    #run_test()
    #show_model_info()
    #model = create_model()
    #show_model_info(model)    
    

