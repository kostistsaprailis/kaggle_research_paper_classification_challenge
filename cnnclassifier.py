import os
import re
import csv
import glob

import tensorflow as tf

import numpy as np
from datetime import datetime

from keras.layers import Dense, Input, GlobalAveragePooling1D
from keras.layers import Conv1D, Embedding, Dropout, Concatenate
from keras.models import Model
from keras.utils import plot_model

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

import keras.backend as K
from keras.models import load_model
from keras.backend import tensorflow_backend

from tensorflow.python.lib.io import file_io

from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

BASE_DIR = '.'
EMBEDDING_DIM = 300
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 240
BATCH_SIZE= 32
DROPOUT_RATE = 0.2
LAYER_SIZE = 256


class NNClassifier(object):
    """
        The implementation of a neural net classifier based on Keras
    """
    def __init__(self):
        train = np.load(os.path.join('npz', 'train.npz'))
        validation = np.load(os.path.join('npz', 'validation.npz'))
        test = np.load(os.path.join('npz', 'test.npz'))
        misc = np.load(os.path.join('npz', 'misc.npz'))
        self.train_data = []
        self.validation = []
        self.test = []
        self.train_data.append(train['train_text'])
        self.train_data.append(train['train_graph'])
        self.train_data.append(train['train_graph_extra'])
        self.train_data.append(train['train_graph_walk'])
        self.train_ids = train['train_ids']
        self.y_train = train['y_train']
        self.validation.append(validation['validation_text'])
        self.validation.append(validation['validation_graph'])
        self.validation.append(validation['validation_graph_extra'])
        self.validation.append(validation['validation_graph_walk'])
        self.y_val = validation['y_val']
        self.validation_ids = validation['validation_ids']
        self.test.append(test['test_text'])
        self.test.append(test['test_graph'])
        self.test.append(test['test_graph_extra'])
        self.test.append(test['test_graph_walk'])
        self.test_ids = test['test_ids'].tolist()
        self.embedding_matrix = misc['embedding_matrix']       
        self.labels = misc['labels'].tolist()
        self.num_words = misc['num_words'].tolist()
        self.max_num_authors = misc['max_num_authors']
        self.num_authors = misc['num_authors']

    def train(self):

        self.timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
        self.model_output_dir = os.path.join(
            'models',
            self.timestamp
        )
        file_io.recursive_create_dir(self.model_output_dir)

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(self.num_words,
                                    EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        graph_embeddings = Embedding(18981,
                                     30,
                                     input_length=238,
                                     trainable=True)
        print('Training model.')

        text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="Title_Abstract_Text_Embeddings")
        embedded_sequences = embedding_layer(text_input)
        embedded_sequences = Dropout(rate=DROPOUT_RATE)(embedded_sequences)
        x1 = Conv1D(128, 3, activation='relu', padding='same')(embedded_sequences)
        x1 = GlobalAveragePooling1D()(x1)
        x5 = Conv1D(128, 7, activation='relu', padding='same')(embedded_sequences)
        x5 = GlobalAveragePooling1D()(x5)
        x6 = Conv1D(128, 11, activation='relu', padding='same')(embedded_sequences)
        x6 = GlobalAveragePooling1D()(x6)
        conc1 = Concatenate()([x1, x5, x6])

        graph_input = Input(shape=(238,), name="Graph_Embeddings")
        graph_embedded = graph_embeddings(graph_input)
        graph_embedded = Dropout(rate=DROPOUT_RATE)(graph_embedded)
        x2 = GlobalAveragePooling1D()(graph_embedded)        

        x3 = Input(shape=(3,), name="In_Out_Degree_Adjacency")

        x4 = Input(shape=(64,), name="DeepWalk")

       
        x = Concatenate()([conc1, x2, x3, x4])
        x = Dense(LAYER_SIZE, activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        preds = Dense(28, activation='softmax')(x)

        inputs = []
        inputs.append(text_input)
        inputs.append(graph_input)
        inputs.append(x3)
        inputs.append(x4)

        self.model = Model(inputs, preds)
        self.model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

        plot_model(self.model, to_file=os.path.join(self.model_output_dir, 'model.png'))

        # self.model.summary()

        self.filepath = os.path.join(
            self.model_output_dir,
            "weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5"
        )

        tensorboard_dir = os.path.join(self.model_output_dir, 'logs')
        tensorboard = TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            batch_size=BATCH_SIZE,
            write_graph=False,
            write_grads=False,
            write_images=False,
            embeddings_freq=1,
            embeddings_layer_names=None)

        checkpoint = ModelCheckpoint(
            self.filepath,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min')

        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=6,
            verbose=0,
            mode='auto')

        reduce_on_plateau = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=1
        )

        clw = []
        for item in self.y_train:
            clw.append(np.argmax(item))
        class_weight = compute_class_weight('balanced',
                                            np.unique(clw),
                                            clw)

        self.model.fit(self.train_data, self.y_train,
                       batch_size=BATCH_SIZE,
                       epochs=40,
                       class_weight=class_weight,
                       callbacks=[checkpoint, early_stopping,tensorboard, reduce_on_plateau],#lr_finder],
                       validation_data=(self.validation, self.y_val))


    def predict(self):
        """
            Predict and get accuracy from the provided test data
        """
        self.load_model()
        y_pred = self.model.predict(self.test)
        return y_pred

    def load_model(self):
        """Load a model from a provided path"""
        try:
            tensorflow_backend.clear_session()
            model_file = self._find_latest_model_path()
            print('Loading model:', model_file)
            self.model = load_model(model_file)
            self.graph = tf.get_default_graph()

        except Exception as e:
            print('Could not load model:', str(e))

    def _find_latest_model_path(self):

        latest_model = None
        max_epoch = 0
        files = [
            file_path
            for file_path
            in glob.iglob(os.path.join(self.model_output_dir, "weights-improvement*.hdf5"), recursive=True)
        ]
        for file in files:
            file = re.sub(self.model_output_dir, '', file)
            if int(file.split('-')[2]) > max_epoch:
                latest_model = self.model_output_dir + file
                max_epoch = int(file.split('-')[2])
        return latest_model

if __name__ == '__main__':
    clf = NNClassifier()
    clf.train()
    y_pred = clf.predict()

    labels = [''] * 28
    for key, value in clf.labels.items():
        labels[value] = key

    with open(os.path.join(clf.model_output_dir, clf.timestamp + '_submission.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        labels.insert(0, "Article")
        writer.writerow(labels)
        print(y_pred.shape)
        for i,test_id in enumerate(clf.test_ids):
            lst = y_pred[i].tolist()
            lst.insert(0, test_id)
            writer.writerow(lst)
