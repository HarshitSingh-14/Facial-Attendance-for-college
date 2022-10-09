from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras.models import load_model
import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import argparse
import pickle

# Construct the argumet parser and parse the argument
from detectfaces_mtcnn.Configurations import get_logger
from src.insightface.alignment.test import M
from training.softmax import SoftMax


class TrainFaceRecogModel:

    def __init__(self, args):

        self.args = args
        self.logger = get_logger()
        # Load the face embeddings
        self.data = pickle.loads(open(args["embeddings"], "rb").read())

    def trainKerasModelForFaceRecognition(self):

        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        # number of classes as per names
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        #interger 1 hot encoding 
        one_hot_encoder = OneHotEncoder(categorical_features = [0])
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array(self.data["embeddings"])

        # Softmax training model arguments
        BATCH_SIZE = 8

        # increase epochds if more number of people
        EPOCHS = 10
        input_shape = embeddings.shape[1]

        # Sofmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # KFold Validation for better acc
        cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
            
            
            modeling = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
            print(modeling.history['acc'])

            history['acc'] += modeling.history['acc']
            history['val_acc'] += modeling.history['val_acc']
            history['loss'] += modeling.history['loss']
            history['val_loss'] += modeling.history['val_loss']

            self.logger.info(modeling.history['acc'])

        # write output
        model.save(self.args['model'])
        f = open(self.args["le"], "wb")
        f.write(pickle.dumps(le))
        f.close()
