# import keras, csv, cv2 and numpy library
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

import csv
import cv2
import numpy as np

def Preprocess(datacsv):
    lines = [] # rows in csv log file
    images = [] # images in data folder recorded from simulator
    measurements = [] # steering angles from csv file

    # read the csv file and append the three camera angle 
    # images to images list and correct the steering angles
    with open(datacsv) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i in range(3):
                source_path = line[i]
                filename = source_path.split('\\')[-1]
                current_path = './data/IMG/' + filename
                image = cv2.imread(current_path)
                images.append(image)
                measurement = float(line[3])
                if i==1:
                    measurement += 0.2
                if i==2:
                    measurement -= 0.2
                    
                measurements.append(measurement)
    # augment images and steering angles by mirroring them     
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurements.append(measurement*-1.0)      
    return np.array(augmented_images), np.array(augmented_measurements)

# train using Nvidia model architecture and save the model
# Note: This fails to save model in windows machine,
# but in ipython notebook it is succesful
def Save_Model(X_train, y_train, epoch=2):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    training = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epoch)
    model.save('model.h5')
# single function for csv input and saved model output 
def Execute(epoch=2):
    X_train, y_train = Preprocess('./data/driving_log.csv')
    Save_Model(X_train, y_train, epoch)
