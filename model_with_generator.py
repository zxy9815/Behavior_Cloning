# general
from __future__ import absolute_import
import os
import sys
import argparse
import csv
import cv2
import numpy as np
from absl import app
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Lambda, Cropping2D
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
    return samples


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = FLAGS.data_dir + '/IMG/' + batch_sample[0].split('/')[-1]
                center_image = preprocess_image(cv2.imread(center_name))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # data augmentation
                augmented_c_image, augmented_c_angle = augment_data(center_image, center_angle)
                images.append(augmented_c_image)
                angles.append(augmented_c_angle)

                # add in left and right cameras' info
                left_name = FLAGS.data_dir + '/IMG/' + batch_sample[1].split('/')[-1]
                left_image = preprocess_image(cv2.imread(left_name))
                right_name = FLAGS.data_dir + '/IMG/' + batch_sample[2].split('/')[-1]
                right_image = preprocess_image(cv2.imread(right_name))
                # create adjusted steering measurements for the side camera images
                correction = 0.3  # this is a parameter to tune
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                # add images and angles to data set
                images.extend([left_image, right_image])
                angles.extend([left_angle, right_angle])

                # data augmentation
                augmented_l_image, augmented_l_angle = augment_data(left_image, left_angle)
                augmented_r_image, augmented_r_angle = augment_data(right_image, right_angle)
                images.extend([augmented_l_image, augmented_r_image])
                angles.extend([augmented_l_angle, augmented_r_angle])

            # trim image to only see section with road
            X = np.array(images)
            y = np.array(angles)

            X, y = sklearn.utils.shuffle(X, y)

            yield X, y

# create model
def create_model():
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Conv2D(6, (5, 5), strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()
    return model


def train(model, train_samples, validation_samples):
    model.compile(optimizer='adam', loss='mse')
    # model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
    val_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

    # train
    history_object = model.fit(x=train_generator,
                               steps_per_epoch=len(train_samples) // FLAGS.batch_size,
                               validation_data=val_generator,
                               validation_steps=len(validation_samples) // FLAGS.batch_size,
                               epochs=FLAGS.epochs,
                               verbose=1)
    model.save('model.h5')

    # print the keys contained in the history object
    print(history_object.history.keys())

    return history_object


def main(_):
    # read data
    samples = read_data(FLAGS.data_dir)

    # data split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # create model
    model = create_model()

    # train
    history_object = train(model, train_samples, validation_samples)

    # display
    display_results(history_object)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directory Parameters:
    parser.add_argument('--data_dir', type=str, default=data_dir,
                        help='Input Data Directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The batch size')

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)

"""
Example:
python model_with_generator.py \
--data_dir ./data/ \
--epochs 5 \
--batch_size 128
"""
