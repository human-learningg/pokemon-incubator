from keras import backend as K
K.set_image_dim_ordering('th')   # ensure our dimension notation matches

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras.datasets import mnist
from keras import utils
import numpy as np
from PIL import Image, ImageOps
import argparse
import math
import sys

import os
import os.path

import glob

from utils import *


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 8, 8), input_shape=(128*8*8,)))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
                        64, 5, 5,
                        border_mode='same',
                        input_shape=(1, 128, 128)))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def train(epochs, BATCH_SIZE, weights=False):
    """
    :param epochs: Train for this many epochs
    :param BATCH_SIZE: Size of minibatch
    :param weights: If True, load weights from file, otherwise train the model from scratch.
    Use this if you have already saved state of the network and want to train it further.
    """
    X_train = load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    discriminator = discriminator_model()
    generator = generator_model()
    if weights:
        generator.load_weights('g.h5', by_name=True)
        discriminator.load_weights('d.h5', by_name=True)
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)
            # print(X_train.shape)
            # print(generated_images.shape)
            # print(image_batch.shape)
            if index % 20 == 0 and epoch % 10 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                # destpath = os.path.normpath(os.getcwd() + "/logo-generated-images/"+str(epoch)+"_"+str(index)+".png")
                destpath = os.path.normpath(os.getcwd() + '/generated/' + str(epoch) + '_' + str(index) + '.png')
                Image.fromarray(image.astype(np.uint8)).save(destpath)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if epoch % 10 == 2:
                generator.save_weights('g.h5', True)
                discriminator.save_weights('d.h5', True)


def clean(image):
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i][j] + image[i+1][j] + image[i][j+1] + image[i-1][j] + image[i][j-1] > 127 * 5:
                image[i][j] = 255
    return image


def generate(BATCH_SIZE):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('g.h5', by_name=True)
    noise = np.zeros((BATCH_SIZE, 100))
    a = np.random.uniform(-1, 1, 100)
    b = np.random.uniform(-1, 1, 100)
    grad = (b - a) / BATCH_SIZE
    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.uniform(-1, 1, 100)
    generated_images = generator.predict(noise, verbose=1)
    #image = combine_images(generated_images)
    print(generated_images.shape)
    for image in generated_images:
        image = image[0]
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save("dirty1.png")
        # Image.fromarray(image.astype(np.uint8)).show()
        clean(image)
        image = Image.fromarray(image.astype(np.uint8))
        # image.show()
        image.save("clean1.png")
