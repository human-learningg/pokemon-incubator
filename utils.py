import os
import glob
import math
import numpy as np
from PIL import Image, ImageOps


def load_data(verbose=False):
    print("Loading data")
    X_train = []
    # paths = glob.glob(os.path.normpath(os.getcwd() + '/training-data/*.png'))
    paths = glob.glob(os.path.normpath(os.getcwd() + '/training-data/originals/*.png'))
    for path in paths:
        if verbose:
            print(path)
        im = Image.open(path)
        im = ImageOps.fit(im, (128, 128), Image.ANTIALIAS)
        im = ImageOps.grayscale(im)
        # im.show()
        im = np.asarray(im)
        X_train.append(im)
    print("Finished loading data")
    return np.array(X_train)


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
