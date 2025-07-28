import pandas as pd
poker = pd.read_csv("https://archive.ics.uci.edu/static/public/158/data.csv")


poker.describe().applymap('{:.2f}'.format) #.describe() formatted it weirdly
poker.head()
#import Dictionaries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#Splitting our data set
X = poker[["S1",	"C1",	"S2",	"C2",	"S3",	"C3", "S4",	"C4",	"S5",	"C5"]]
y = poker["CLASS"]

#Splitting our training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#Scaling Data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Grid Search, had to reduce it to the best values as compile time takes too long
pipe = Pipeline(steps=[('classifier', 'passthrough')])

seed=42
param_grid = [
    {  # Null Model
        'classifier': [DummyClassifier(strategy='most_frequent', random_state = seed)]
    },
     { # KNN
        'classifier': [DecisionTreeClassifier(random_state = seed)],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]

    },
     { # Neural Net
        'classifier': [MLPClassifier(random_state=seed, max_iter = 1000, early_stopping = True)],
        'classifier__hidden_layer_sizes': [(50,50)], # [(50,50), (50, 100, 50), (100,)]
        'classifier__activation': ['tanh'],
        'classifier__solver': ['adam'],
        'classifier__alpha': [1e-6] #[0.01, 1e-4,1e-6]
    }
]

#other model
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1, scoring = "accuracy")
grid_search.fit(X_train, y_train)

print("tuned hyperparameters and best model:",grid_search.best_params_)
print("accuracy :",grid_search.best_score_)
