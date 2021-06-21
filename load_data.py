
import os
import sys
import seaborn as sns
import matplotlib.image as mpimg

import graphviz
from dtreeviz.trees import dtreeviz
from sklearn.metrics import classification_report
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from treelib import Node, Tree
from tqdm.auto import tqdm, trange
from imblearn.pipeline import make_pipeline
import random

import utils

plt.rcParams['figure.figsize'] = (8, 4)

def load_data(log_transform=False):
    tracks = utils.load('data/fma_metadata/tracks.csv')
    genres = utils.load('data/fma_metadata/genres.csv')
    echonest = utils.load('data/fma_metadata/echonest.csv')
    
    # 15 = Electronic
    parent_genre = 15

    selected_genres = genres.index[genres["parent"] == parent_genre].tolist()
    selected_genre_names = {genres["title"].loc[idx]: idx for idx in selected_genres}
    class_names = list(selected_genre_names.keys())

    pbar = tqdm(tracks['track'].iterrows(), total = len(tracks))
    selected_tracks = [(ri, [1. if g in row['genres_all'] else 0. for g in selected_genres])
                       for ri, row in pbar if np.any([g in row['genres_all'] for g in selected_genres])]
    
    #  
    indices = {ri: np.argmax(g) for ri, g in selected_tracks}
    echonest_indices = echonest['echonest', 'temporal_features'].index.to_numpy()
    temporal_features = echonest['echonest', 'temporal_features'].to_numpy()
    data = [(temporal_features[e_i], indices[i]) for e_i, i in enumerate(echonest_indices) if i in indices]
    X, y = np.stack([np.array(d[0]) for d in data]), np.array([d[1] for d in data])

    # dim reduction
    new_dim = 50
    ratio = X.shape[-1]/new_dim

    new_X = []
    for i in range(len(X)):
        row = []
        for i2 in range(new_dim):
            v = i2*ratio
            a, b = int(v), int(v + 0.5)
            fr = v - a
            value = (X[i, a]*(1.-fr)+X[i, b]*fr) 
            row.append(value)
        new_X.append(np.array(row))
    X = np.stack(new_X)

    t_f = int(len(X)*0.8)
    X_train, X_test = X[:t_f], X[t_f:]
    y_train, y_test = y[:t_f], y[t_f:]
    
    if log_transform:
        X_train = np.log(X_train-np.min(X_train, axis=0)+1)
        X_test = np.log(X_test-np.min(X_test, axis=0)+1)
        
    return X_train, y_train, X_test, y_test