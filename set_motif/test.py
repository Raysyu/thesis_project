import copy
import datetime
import logging
import os
import pickle
from multiprocessing import Pool
import bz2
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import psutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


from set_mlp import SET_MLP, Relu,MSE,Softmax,CrossEntropy
from utils.monitor import Monitor

from train_utils import sample_weights_and_metrics

from sklearn.datasets import fetch_california_housing,load_diabetes,load_wine


def wine_data():
    data = load_wine()
    x = data.data
    y = data.target
    return x, y

def wine_train(set_params):
    sum_training_time = 0
    n_hidden_neurons_layer = set_params['n_hidden_neurons_layer']
    epsilon = set_params['epsilon']
    zeta = set_params['zeta']
    batch_size = set_params['batch_size']
    dropout_rate = set_params['dropout_rate']
    learning_rate = set_params['learning_rate']
    momentum = set_params['momentum']
    weight_decay = set_params['weight_decay']

    x,y = wine_data()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.8)
    set_mlp = SET_MLP((x_train.shape[1], n_hidden_neurons_layer, n_hidden_neurons_layer,
                       y_train.shape), (Relu, Relu, Softmax), epsilon=epsilon)
    start_time = datetime.datetime.now()
    set_metrics = set_mlp.fit(x_train, y_train, x_test, y_test, loss=CrossEntropy, epochs=15,
                              batch_size=batch_size, learning_rate=learning_rate,
                              momentum=momentum, weight_decay=weight_decay, zeta=zeta, dropout_rate=dropout_rate,
                              testing=True,
                              save_filename="", monitor=False)


    # After every epoch we store all weight layers to do feature selection and topology comparison
    evolved_weights = set_mlp.weights_evolution

    dt = datetime.datetime.now() - start_time

    step_time = datetime.datetime.now() - start_time
    print("\nTotal training time: ", step_time)
    sum_training_time += step_time


set_params = {'n_hidden_neurons_layer': 3000,
                  'epsilon': 13,  # set the sparsity level\
                  'n_training_epochs': 10 ,
                  'zeta': 0.3,  # in [0..1]. Percentage of unimportant connections to be removed and replaced
                  'batch_size': 10, 'dropout_rate': 0, 'learning_rate': 0.05, 'momentum': 0.9, 'weight_decay': 0.0002,
              }
wine_train(set_params)




