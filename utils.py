import os
import glob
import math
import argparse
import scipy.misc
import numpy as np
from PIL import Image, ImageOps


IMAGE_DIMENSION = 128


def load_data():
    print("Loading data")
    X_train = []
    paths = glob.glob(os.path.normpath(os.getcwd() + '/training-data/*.png'))
    for path in paths:
        im = Image.open(path)
        im = remove_alpha(im)
        im = ImageOps.fit(im, (IMAGE_DIMENSION, IMAGE_DIMENSION), Image.ANTIALIAS)
        # im = ImageOps.grayscale(im)
        # im.show()
        im = np.asarray(im)
        X_train.append(im)
    print("Finished loading data")
    return np.array(X_train)


def remove_alpha(img):
    img.load()

    bg = Image.new('RGB', img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])

    return bg


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    return (images+1.)/2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-interval', type=int, default=50)
    parser.add_argument('--load-saved', type=bool, default=True)
    return parser.parse_args()
