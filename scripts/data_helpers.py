import numpy as np
import cv2
import time

from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from itertools import product

def create_pairs(data, ratio_0_1=1, discard_rest=True):
    """
    Creates data pairs
    :param data: 2D list containing data for each class
    :param ratio_0_1: Ratio between same and different pairs.
    The higher the ratio the more same pairs.
    :param discard_rest: Determines if uncombined data will be paired in its own class
    to retain as much data as possible
    :return: Returns same and different pairs as well as the rest of the uncombined data
    """

    # make a copy and shuffle the data of each class
    grouped_data = data.copy()
    np.random.seed(42)
    for data in grouped_data:
        np.random.shuffle(data)

    # create pairs of same classes
    same_pairs = []
    for i, data in enumerate(grouped_data):
        num_same = int((ratio_0_1 * len(data)) / (ratio_0_1 + 1))
        num_same = num_same if num_same % 2 == 0 else num_same - 1
        same_data, grouped_data[i] = data[:num_same], data[num_same:]
        same_pairs.extend(list(zip(same_data[:num_same // 2], same_data[num_same // 2:])))

    # create pairs of different classes
    diff_pairs = []
    while len([i for i, data in enumerate(grouped_data) if len(data) != 0]) > 1:
        smallest_list_idx = sorted([i for i, data in enumerate(grouped_data) if len(data) != 0],
                                   key=lambda i: len(grouped_data[i]))[0]
        idxs = [i for i, data in enumerate(grouped_data) if len(data) != 0 and i != smallest_list_idx]
        idxs = np.random.choice(idxs, min(len(grouped_data[smallest_list_idx]), len(idxs)), replace=False)
        diff_pairs.extend([[grouped_data[smallest_list_idx].pop(), grouped_data[idx].pop()] for idx in idxs])

    # unused data can be paired with its own class to don't waste data, but imbalance could occur
    if not discard_rest:
        for i, data in enumerate(grouped_data):
            if len(data) != 0:
                num_same = len(data) if len(data) % 2 == 0 else len(data) - 1
                same_data, grouped_data[i] = data[:num_same], data[num_same:]
                same_pairs.extend(list(zip(same_data[:num_same // 2], data[num_same // 2:])))

    return same_pairs, diff_pairs, grouped_data


def make_square(img, color=None):
    """
    Creates squared images by increasing the canvas and centering the image
    :param img: The image to resize
    :param color: Color of the background canvas to use.
    If None the borders of the image will be replicated
    :return: Returns square image
    """

    max_dim = max(img.shape[:2])
    delta_w = max_dim - img.shape[1]
    delta_h = max_dim - img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    if color is None:
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    else:
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def create_img_pairs(file_pairs, mode, target_size):
    """
    Loads data and preprocesses it
    :param file_pairs: File pairs to obtain images
    :param mode: In test mode, data will be normalized immediately.
    In train mode, this is not the case because of following augmentations.
    :param target_size: Size to reshape images to
    :return: Returns image pairs
    """

    img_pairs = []
    i = 0

    for file_pair in file_pairs:
        img_pair = [cv2.imread(img_file, -1) for img_file in file_pair]
        img_pair = [make_square(img) for img in img_pair]
        img_pair = [cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC) for img in img_pair]

        if mode == 'test':
            img_pair = [img / 255.0 for img in img_pair]
            img_pair = [img.astype('float32') for img in img_pair]

        img_pairs.append(img_pair)

        if i % 100 == 0:
            if i == 0:
                start = time.time()
            else:
                end = time.time()
                print('Progressed time: {:.2f} sec - ETA: {:.2f} sec'.format(
                    end - start, (len(file_pairs) - i) * ((end - start) / i)))

        i += 1

    return img_pairs


def split_imgs(img_pairs):
    """
    Transforms image pairs into correct shape for training
    :param img_pairs:
    :return: Returns data in necessary shape
    """

    splitted = np.split(img_pairs, 2, axis=1)
    splitted[0] = splitted[0][:, 0, :, :, :]
    splitted[1] = splitted[1][:, 0, :, :, :]

    return splitted


def augment_img(img, flips, image_gen, seed, scale_range=(0.7, 1.3), tx=10, ty=10):
    np.random.seed(seed)

    flip = flips[np.random.choice(len(flips))]

    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    transform_parameters = {
        'tx': np.random.uniform(-tx, tx),
        'ty': np.random.uniform(-ty, ty),
        'flip_horizontal': flip[0],
        'flip_vertical': flip[1],
        'zx': scale_factor,
        'zy': scale_factor
    }

    return image_gen.apply_transform(img, transform_parameters)


class DataGenerator(Sequence):
    """
    Generates data while training
    """

    def __init__(self, img_pairs, labels, augment, equalized_hist, batch_size=32, shuffle=True):
        self.img_pairs = img_pairs
        self.labels = labels
        self.augment = augment
        self.equalized_hist = equalized_hist
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flips = list(product([True, False], [True, False]))
        self.image_gen = ImageDataGenerator()
        self.epoch = 0
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.img_pairs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch
        idxs = self.idxs[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x = self.__data_generation(idxs, (self.img_pairs[k] for k in idxs))
        y = [self.labels[k] for k in idxs]

        return x, y

    def on_epoch_end(self):
        """Updates indices after each epoch"""
        self.epoch += 1

        self.idxs = np.arange(len(self.img_pairs))
        if self.shuffle:
            np.random.seed(self.epoch)
            np.random.shuffle(self.idxs)

    def __data_generation(self, idxs, batch_pairs):
        """Augments/Preprocess image pairs"""
        img_pairs = []
        for idx, img_pair in zip(idxs, batch_pairs):
            if self.augment:
                seed = (self.epoch * idx) % (2 ** 32 - 1)
                img_pair = [augment_img(img, self.flips, self.image_gen, seed=seed + i)
                            for i, img in enumerate(img_pair)]

            # Normalize image
            img_pair = [img / 255.0 for img in img_pair]
            img_pair = [img.astype('float32') for img in img_pair]

            img_pairs.append(img_pair)

        return split_imgs(np.array(img_pairs))