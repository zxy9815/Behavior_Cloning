# general
import os
import sys
import argparse
import csv
import cv2
import numpy as np
from absl import app
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, Lambda, Cropping2D
from utils import preprocess_image, augment_data, display_results

FLAGS = None
data_dir = './data'


# read data
def read_data(data_path):
    samples = []
    log_path = os.path.join(os.path.abspath(data_path), 'driving_log.csv')
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    images = []
    angles = []
    for line in samples:
        filename = FLAGS.data_dir + '/IMG/' + line[0].split('/')[-1]
        image = cv2.imread(filename)
        angle = float(line[3])
        image = preprocess_image(image)
        images.append(image)
        angles.append(angle)
        # data augmentation
        augmented_image, augmented_angle = augment_data(image, angle)
        images.append(augmented_image)
        angles.append(augmented_angle)
    return np.array(images), np.array(angles)


# create model
def create_model():
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(6, (5, 5), strides=(2, 2), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPool2D(strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model


def train(model, X_train, y_train):
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=FLAGS.epochs, verbose=1)
    model.save('model.h5')

    # print the keys contained in the history object
    print(history_object.history.keys())

    return history_object


def main(_):
    # read data
    X_train, y_train = read_data(FLAGS.data_dir)

    print(X_train.shape, y_train.shape)

    # create model
    model = create_model()

    # train
    history_object = train(model, X_train, y_train)

    # display
    display_results(history_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directory Parameters:
    parser.add_argument('--data_dir', type=str, default=data_dir,
                        help='Input Data Directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of epochs')

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)

"""
Example:
python model.py \
--data_dir ./data/ \
--epochs 5
"""
