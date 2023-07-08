#======================================================#
#== Sameh Algharabli - CNG 1530 -- Assignment3 --  ==#
#======================================================#

import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math as m
import pandas as pd

import random
random.seed(11)
np.random.seed(17)

# Reading the datset
dataset = pd.read_csv('breast-cancer-wisconsin.data', sep=',', header=None)

# Changing all '?' value with nan
dataset = dataset.replace("?", np.nan)
print("===============================================\n")
print("PREPROCESSING\n")
print("Before removing missing values, the data set has %d rows"%(len(dataset)))
print("The data has %d rows with missing values"%(dataset.isna().any(axis=1).sum()))

# Dropping all rows with nan value
dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

print("After removing missing values, the data set has %d rows"%(len(dataset)))
print("===============================================\n")

# Taking the features (from attribute 2 to 10), and the labels (attribute 11)
features = dataset[dataset.columns[1:-1]]
labels = dataset[dataset.columns[-1]]

# Changing the labels from 2 and 4 to 0 & 1 respectively
labels = labels.replace([2], 0)
labels = labels.replace([4], 1)

# END OF PREPROCESSING
#----------------------------------------------------------#

# This function displays the result of each model
def display_result(algo, model):
    print(algo + " Results: ")
    print("---------------")
    print("Mean validation score: ", model.cv_results_["mean_test_score"])
    print("Std of validation score: ", model.cv_results_["std_test_score"])
    print("Best mean validation score: ", model.best_score_)
    print("Best parameters of the model: ", model.best_params_)



# Using the repeated statified k-fold cross validation with 10 splits and 5 repeats
cross_validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=np.random.randint(1, 1000))


# Splitting the data into training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20)

print("Splitting the dataset into training & testing\n")
print("Training : 75% --> ",features_train.shape)
print("Testing  : 25% --> ",features_test.shape)
print("===============================================\n")

#-------------------------------------------------------------------------------#
print("Modeling \n")

# A list of the three algorithms used for classification
algorthims = ["SVM", "KNN", "Naive Bayes"]

# The different parameters of each algorithms
svm_parameter_grid = {"C": [0.1, 0.5],
              "kernel": ["poly", "rbf"]
}
knn_parameter_grid = {"metric": ["cosine", "euclidean"],
                    "n_neighbors": [2,  3]
                    }

NB_parameter_grid = {"var_smoothing": [0.0000001, 0.00000001, 0.000000001]}

#-------------------
# A loop to go through the list of algorithms
for algorthim in algorthims:
    if algorthim == "SVM":
        model = SVC()
        grid = svm_parameter_grid
    elif algorthim == "KNN":
        model = KNeighborsClassifier()
        grid = knn_parameter_grid
    else:
        model = GaussianNB()
        grid = NB_parameter_grid

    # Training using GridSearch and cross validation to find best parameters according the highest accuracy
    print("Training " + algorthim + "...")
    trained_model = GridSearchCV(model, grid, scoring="accuracy", cv=cross_validation, verbose=False, refit=True)
    trained_model.fit(features_train, labels_train) # Fitting the model
    display_result(algorthim, trained_model) # Printing the training result

    predictions = trained_model.predict(features_test) # getting predictions
    test_accuracy = accuracy_score(labels_test, predictions) # evaluation the prediction
    print("Testing accuracy score: ", test_accuracy) # printing the f1_score of the prediction
    print("----------------------------------------------------------------\n")

print("===============================================\n")
