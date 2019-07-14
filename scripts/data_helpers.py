import numpy as np
import cv2
import time

from os import makedirs
from os.path import dirname, isdir, basename
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

    
def preprocess_img(image, target_size, canvas_color=(255, 255, 255), normalize=True):
    """
    Preprocesses image for training
    :param image: Image to process
    :param target_size: Output size to resize image to
    :param canvas_color: Color of the square canvas the image will be in
    :param normalize: Determines if image will be normalized
    :return: Returns preprocessed image
    """
    
    img = image.copy()
    img = make_square(img, color=canvas_color)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

    if normalize:
        img = (img / 255.0).astype('float32')

    return img


def split_imgs(img_pairs):
    """
    Transforms image pairs into correct shape for training
    :param img_pairs: Paired images
    :return: Returns data in necessary shape
    """

    splitted = np.split(img_pairs, 2, axis=1)
    splitted[0] = splitted[0][:, 0, :, :, :]
    splitted[1] = splitted[1][:, 0, :, :, :]

    return splitted


def augment_img(img, image_gen, seed, scale_range=(0.7, 1.3), tx=10, ty=10):
    """
    Augment images by flipping horizontally and vertically, zooming and shifting in x and y-direction
    :para img: Image to augment
    :param image_gen: Image generator to used
    :param seed: Set different seeds for deterministic but still random behavior
    :param scale_range: Zoom factor
    :param tx: Amount of pixels to shift in x direction
    :param ty: Amount of pixels to shift in y direction
    """
    
    flips = list(product([True, False], [True, False]))
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

    def __init__(self, img_pairs, labels, augment, batch_size=32, shuffle=True):
        self.img_pairs = img_pairs
        self.labels = labels
        self.augment = augment
        self.batch_size = batch_size
        self.shuffle = shuffle
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
                img_pair = [augment_img(img, self.image_gen, seed=seed + i)
                            for i, img in enumerate(img_pair)]

            # Normalize image
            img_pair = [img / 255.0 for img in img_pair]
            img_pair = [img.astype('float32') for img in img_pair]

            img_pairs.append(img_pair)

        return split_imgs(np.array(img_pairs))
    
    
def create_metadata(file_path, files, classes):
    """
    Creates tab seperated metadata file to visualize classes in embeddings
    :param file_path: Path of file to be saved
    :param files: Files to visualize
    :param classes: Corresponding classes of files (in sequential order)
    :return:
    """
    if not isdir(dirname(file_path)):
        makedirs(dirname(file_path))

    with open(file_path, 'w') as fn:
        fn.write('Filename\tClass\n')
        for file, vis_class in zip(files, classes):
            fn.write('{}\t{}\n'.format(basename(file), vis_class))
            
            
            
            