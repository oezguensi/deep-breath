import numpy as np
import keras.backend as K

from glob import glob
from os.path import join
from os import remove
from keras import Input
from keras.layers import Dense, Lambda, GlobalAveragePooling2D
from keras.models import Model
from keras.applications import MobileNetV2
from keras.callbacks import Callback


def create_mobile_net_encoder(input_shape, dense_layers=None, mobilenet_width=1):
    """
    Retrieves the mobilenet without its last layer
    :param input_shape: Shape of the images to input into the network
    :param dense_layers: A (optional) list of numbers determining the dimension of additional dense layers
    :param mobilenet_width: The width of the mobilenet determining the number of filters per layer
    :return: Mobilenet
    """

    mobilenet = MobileNetV2(input_shape=input_shape, weights=None, include_top=False, alpha=mobilenet_width)
    output = mobilenet.output
    output = GlobalAveragePooling2D()(output)
    if dense_layers:
        for layer in dense_layers:
            output = Dense(layer, activation='relu')(output)

    return Model(inputs=mobilenet.input, outputs=output)


def euclidean_distance(v, w):
    """
    Implementation of the euclidean distance
    :param v: First vector
    :param w: Second vector
    :return: Euclidean distance function
    """

    assert len(v) == len(w), 'Both vectors must have the same dimensions'

    return K.sqrt(K.maximum(K.sum(K.square(v - w), axis=1, keepdims=True), K.epsilon()))


def create_siamese_model(encoder, distance_func):
    """
    Creates a siamese model by attaching two encoders and calculating the distance of their encodings
    :param encoder: The encoder to use for each side of the Siamese network
    :param distance_func: Distance function to use
    :return: Siamese model
    """

    input_1 = Input(encoder.input_shape[1:], name='first_input')
    input_2 = Input(encoder.input_shape[1:], name='second_input')

    encoding_1 = encoder(input_1)
    encoding_2 = encoder(input_2)


    distance = Lambda(distance_func, name='distance')([encoding_1, encoding_2])

    return Model(inputs=[input_1, input_2], outputs=distance)


def contrastive_loss(y_true, distance, margin=1.0):
    """
    Implementation of the contrastive loss from http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param y_true: The ground truth labels
    :param distance: The calculated distance metric
    :param margin: Margin that sets the upper bound
    :return: Loss function
    """

    distance_square = K.square(distance)
    margin_dist_square = K.square(K.maximum(0.0, margin - distance))

    return (1 - y_true) * 0.5 * distance_square + y_true * 0.5 * margin_dist_square


class CustomModelCheckpoint(Callback):
    """
    Saves the last and best weights as well as the complete model according to the monitored metric
    """

    def __init__(self, dirpath, monitor='val_loss', verbose=0, save_weights_only=False, mode='auto', period=1):

        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.dirpath = dirpath
        self.weights_path = join(
            dirpath, '{save}_{file}_epoch-{epoch:04d}_loss-{loss:.6f}_val_loss-{val_loss:.6f}.hdf5')
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f' % (
                            epoch + 1, self.monitor, self.best, current))
                    self.best = current

                    for ckpt_file in glob(join(self.dirpath, 'best_weights*')):
                        remove(ckpt_file)
                    self.model.save_weights(
                        self.weights_path.format(
                            save='best', file='weights', epoch=epoch + 1, **logs))

                    if not self.save_weights_only:
                        for ckpt_file in glob(join(self.dirpath, 'best_model*')):
                            remove(ckpt_file)
                        self.model.save(
                            self.weights_path.format(
                                save='best', file='model', epoch=epoch + 1, **logs))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f' %
                              (epoch + 1, self.monitor, self.best))

            for ckpt_file in glob(join(self.dirpath, 'last_weights*')):
                remove(ckpt_file)
            self.model.save_weights(
                self.weights_path.format(
                    save='last', file='weights', epoch=epoch + 1, **logs))
            if not self.save_weights_only:
                for ckpt_file in glob(join(self.dirpath, 'last_model*')):
                    remove(ckpt_file)
                self.model.save(self.weights_path.format(
                    save='last', file='model', epoch=epoch + 1, **logs))