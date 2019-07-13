import numpy as np
import keras.backend as K
import os

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


def euclidean_distance(vectors):
    """
    Implementation of the euclidean distance
    :param v: First vector
    :param w: Second vector
    :return: Euclidean distance function
    """
    
    assert vectors[0].shape[1] == vectors[1].shape[1], 'Both vectors must have the same dimensions'

    return K.sqrt(K.maximum(K.sum(K.square(vectors[0] - vectors[1]), axis=1, keepdims=True), K.epsilon()))


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


def siamese_accuracy(y_true, distance):
    return K.mean(K.equal(y_true, K.cast(distance > 0.5, y_true.dtype)))


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


class CustomTensorBoard(Callback):
    """TensorBoard basic visualizations.
    [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
    is a visualization tool provided with TensorFlow.
    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.
    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```sh
    tensorboard --logdir=/full_path_to_your_logs
    ```
    When using a backend other than TensorFlow, TensorBoard will still work
    (if you have TensorFlow installed), but the only feature available will
    be the display of the losses and metrics plots.
    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation
            and weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        batch_size: size of batch of inputs to feed to the network
            for histograms computation.
        write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard.
            `histogram_freq` must be greater than 0.
        write_images: whether to write model weights to visualize as
            image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved. If set to 0, embeddings won't be computed.
            Data to be visualized in TensorBoard's Embedding tab must be passed
            as `embeddings_data`.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/guide/embedding#metadata)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in
            `embeddings_layer_names`. Numpy array (if the model has a single
            input) or list of Numpy arrays (if the model has multiple inputs).
            Learn [more about embeddings](
            https://www.tensorflow.org/guide/embedding).
        update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
            the losses and metrics to TensorBoard after each batch. The same
            applies for `'epoch'`. If using an integer, let's say `10000`,
            the callback will write the metrics and losses to TensorBoard every
            10000 samples. Note that writing too frequently to TensorBoard
            can slow down your training.
    """

    def __init__(self, log_dir='./logs',
                 batch_size=32,
                 encoder=None,
                 embeddings_freq=0,
                 embeddings_metadata=None,
                 embeddings_data=None,
                 update_freq='epoch'):
        super(CustomTensorBoard, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to '
                              'use TensorBoard.')

        if K.backend() != 'tensorflow':
            if embeddings_freq != 0:
                warnings.warn('You are not using the TensorFlow backend. '
                              'embeddings_freq was set to 0')
                embeddings_freq = 0

        self.log_dir = log_dir
        self.encoder = encoder
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata or {}
        self.batch_size = batch_size
        self.embeddings_data = embeddings_data
        if update_freq == 'batch':
            # It is the same as writing as frequently as possible.
            self.update_freq = 1
        else:
            self.update_freq = update_freq
        self.samples_seen = 0
        self.samples_seen_at_last_write = 0

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

        self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq and self.embeddings_data is not None:
            # self.embeddings_data = standardize_input_data(self.embeddings_data,
            #                                               model.input_names)

            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            # for layer in self.model.layers:
            #     if layer.name in embeddings_layer_names:
            embedding_input = self.encoder.layers[-1].output  
            embedding_size = np.prod(embedding_input.shape[1:])
            embedding_input = tf.reshape(embedding_input,
                                         (step, int(embedding_size)))
            shape = (self.embeddings_data.shape[0], int(embedding_size))
            embedding = tf.Variable(tf.zeros(shape),
                                    name=self.encoder.layers[-1].name + '_embedding')
            embeddings_vars[self.encoder.layers[-1].name] = embedding
            batch = tf.assign(embedding[batch_id:batch_id + step],
                              embedding_input)
            self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            # TODO
            # if not isinstance(self.embeddings_metadata, str):
            #     embeddings_metadata = self.embeddings_metadata
            # else:
            #     embeddings_metadata = {layer_name: self.embeddings_metadata
            #                            for layer_name in embeddings_vars.keys()}

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                # TODO
                # if layer_name in embeddings_metadata:
                #     embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data.shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    # TODO get_input_at(1)
                    feed_dict = {self.encoder.get_input_at(0): embeddings_data[batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir,
                                                 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        self._write_logs(logs, index)

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

    def on_batch_end(self, batch, logs=None):
        if self.update_freq != 'epoch':
            self.samples_seen += logs['size']
            samples_seen_since = self.samples_seen - self.samples_seen_at_last_write
            if samples_seen_since >= self.update_freq:
                self._write_logs(logs, self.samples_seen)
                self.samples_seen_at_last_write = self.samples_seen