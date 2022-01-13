"""

Course:     20942 - Intro to computational learning
Institute:  Open University of Israel
Assignment: Maman 13
Name:       Nina Verzun
ID:         304680473
Date:       Jan 8 2021

"""
import time

import smo_svm_part_A as smo
import numpy as np
import seaborn as sns
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, recall_score

""" Q4 Part B.a """


def q2_part_b_section_a():
    iris_in = sns.load_dataset("iris")
    sns.pairplot(iris_in, hue="species")
    plt.show()


def q2_part_b_section_b():
    best_c = 0
    best_acc = 0
    for c_val in [1.0, 10.0, 100.0, 1000.0]:
        print(f"========================== Training with C={c_val} ==========================")
        linear_models = init(X_train, y_train, c_val, smo.SMOsvm.Kernel.LINEAR)
        linear_models = train(linear_models, c_val)
        y_val_prediction = predict(linear_models, X_val)
        accuracy = get_accuracy(y_val, y_val_prediction)
        acc = max(accuracy)
        best_c = c_val if acc > best_acc else best_c
        best_acc = acc if acc > best_acc else best_acc
        print(f'Accuracy for C={c_val}: {str(accuracy)}')

    return best_c


def q2_part_b_section_e_tune_c():
    best_c = 0
    best_acc = 0
    for c_val in [1.0, 10.0, 100.0, 1000.0]:
        print(f"========================== Training with C={c_val} ==========================")
        rbf = init(X_train, y_train, c_val, smo.SMOsvm.Kernel.RBF)
        rbf = train(rbf, c_val)
        y_val_prediction = predict(rbf, X_val)
        accuracy = get_accuracy(y_val, y_val_prediction)
        acc = max(accuracy)
        best_c = c_val if acc > best_acc else best_c
        best_acc = acc if acc > best_acc else best_acc
        print(f'Accuracy for C={c_val}: {str(accuracy)}')
    return best_c


def plot_confusion(y_test_in, y_fit_in, class_a, class_b, class_c):
    confusion_mat = confusion_matrix(y_test_in, y_fit_in)
    sns.heatmap(confusion_mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=[class_a, class_b, class_c],
                yticklabels=[class_a, class_b, class_c])
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    plt.show()
    return confusion_mat


def get_accuracy(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    ACC, TNR, TPR = calculate_accuracy(confusion_mat)
    return ACC


def show_metrics(y_true, y_pred):
    order = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    confusion_mat = plot_confusion(y_true, y_pred, order[0], order[1], order[2])
    ACC, TNR, TPR = calculate_accuracy(confusion_mat)
    print(f"========================== Confusion Matrix: ==========================")
    print(confusion_mat)
    print(f'Sensitivity: {TPR}')
    print(f'Selectivity: {TNR}')
    print(f'Accuracy: {ACC}')


def calculate_accuracy(confusion_mat):
    FP = confusion_mat.sum(axis=0) - np.diag(confusion_mat)
    FN = confusion_mat.sum(axis=1) - np.diag(confusion_mat)
    TP = np.diag(confusion_mat)
    TN = confusion_mat.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    return ACC, TNR, TPR


# initiate models with updated labels
def init(x_train, y_train, C, kernel):
    labels = np.unique(y_train)
    models = []
    for label in labels:
        y_train_one_vs_all = np.array([1 if y == label else 0 for y in y_train])
        l = len(y_train)
        initial_alphas = np.zeros(l)
        initial_b = 0.0
        # Instantiate model
        model = smo.SMOsvm(x_train, y_train_one_vs_all, C, kernel, initial_alphas, initial_b, np.zeros(l))
        # Initialize error cache
        initial_error = model.decision_function(model.X, model.X) - model.y
        model.errors = initial_error
        models.append(model)
    return models


# train models
def train(models, c_in):
    tic = time.perf_counter()
    for model in models:
        model.train(error_tolerance, c_in)
    toc = time.perf_counter()
    smo.show_timer(tic, toc)
    return models


def predict(models, x_test):
    evaluations = []
    for model in models:
        u = model.decision_function(model.X, x_test)
        evaluations.append(u)
    y_pred = np.argmax(np.array(evaluations), axis=0)
    return y_pred


print(f"========================== Q4 part B section A ==========================")
print("========================== Train linear models ==========================")
error_tolerance = 0.01  # error tolerance
alpha_tolerance = 0.01  # alpha tolerance

iris = datasets.load_iris()
X = iris.data
y = iris.target
q2_part_b_section_a()

print(f"========================== Q4 part B section B ==========================")
# Split the Iris data into training, validation and test sets
# Train your three binary SVMs with a linear kernel and use the validation set to tune the penalty parameter 𝐶.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

best_c = q2_part_b_section_b();

print(f"========================== Best C={best_c} for Linear Model ==========================")
print(f"\n\n========================== Q4 part B section C ==========================")
linear_models = init(X_train, y_train, best_c, smo.SMOsvm.Kernel.LINEAR)
linear_models = train(linear_models, best_c)
y_test_prediction = predict(linear_models, X_test)
show_metrics(y_test, y_test_prediction)

print(f"========================== Q4 part B section E ==========================")
best_c = q2_part_b_section_e_tune_c();

print(f"========================== Best C={best_c} for RBF Model ==========================")
rbf = init(X_train, y_train, best_c, smo.SMOsvm.Kernel.RBF)
rbf = train(rbf, best_c)
y_test_prediction = predict(rbf, X_test)
show_metrics(y_test, y_test_prediction)

