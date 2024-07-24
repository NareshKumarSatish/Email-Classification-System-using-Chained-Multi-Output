import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from numpy import *
import random
num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

class RandomForest:
    def __init__(self, name, embeddings, class_col):
        self.name = name
        self.embeddings = embeddings
        self.class_col = class_col
        self.mdl = RandomForestClassifier()  # Initialize the model

    def train(self, data):
        X_train = data.get_X_train(self.class_col)
        y_train = data.get_type_y_train(self.class_col)
        self.mdl = self.mdl.fit(X_train, y_train)

    def predict(self, data):
        X_test = data.get_X_test(self.class_col)
        self.predictions = self.mdl.predict(X_test)
