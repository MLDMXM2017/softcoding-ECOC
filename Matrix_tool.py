"""
this method define some tool kit to checkout matrix
or get data subset from data set
"""

import copy
import numpy as np
import math
from Distance import euclidean_distance


def _get_data_from_col(data, label, col, index):
    """
    to get data subset form a col, where the value is not zero
    :param data: data set
    :param label: label corresponding to data
    :param col: the col we want to get data subset
    :param index: the index for matrix
    :return: data subset and corresponding labels
    """
    data_result = None
    cla_result = None
    for i in range(len(col)):
        if col[i] != 0:
            d = np.array([data[k] for k in range(len(label)) if label[k] == _get_key(index, i)])
            c = np.ones(len(d)) * col[i]
            if d.shape[0] > 0 and d.shape[1] > 0:
                if data_result is None:
                    data_result = copy.copy(d)
                    cla_result = copy.copy(c)
                else:
                    data_result = np.vstack((data_result, d))
                    cla_result = np.hstack((cla_result, c))
    return data_result, cla_result


def _closet_vector(vector, matrix, distance=euclidean_distance, weights=None):
    """
    find the closet coding vector in matrix
    :param vector: a predicted vector
    :param matrix: coding matrix
    :param distance: a callable object to calculate distance
    :param weights: the weights for each feature
    :return: the index corresponding to closet coding vector
    """
    d = np.inf
    index = None
    for i in range(matrix.shape[0]):
        if distance(vector, matrix[i], weights) < d:
            d = distance(vector, matrix[i], weights)
            index = i
    return index


def _get_key(dictionary, value):
    for i in dictionary:
        if dictionary[i] == value:
            return i
    return 'Unknown'

def _transform(labels, index):
    values = np.zeros(len(labels))
    for i in range(len(values)):
        if labels[i] not in index:
            values[i] = len(index)
            continue
        values[i] = index[labels[i]]
    return values.astype('int')

def _exist_same_row(matrix):
    """
    to checkout whether there are same rows in a matrix
    :param matrix: coding matrix
    :return: true or false
    """
    row_count = matrix.shape[0]
    for i in range(row_count):
        for j in range(i+1, row_count):
            if np.all([matrix[i] == matrix[j]]) or np.all([matrix[i] == -matrix[j]]):
                return True
    return False


def _exist_same_col(matrix):
    """
    to checkout whether there are same cols in a matrix
    :param matrix: coding matrix
    :return: true or false
    """
    col_count = matrix.shape[1]
    for i in range(col_count):
        for j in range(i+1, col_count):
            if np.all([matrix[:, i] == matrix[:, j]]) or np.all([matrix[:, i] == -matrix[:, j]]):
                return True
    return False


def _exist_two_class(matrix):
    """
    to ensure all cols in coding matrix have 1 and -1
    :param matrix: coding matrix
    :return: true or false
    """
    col_count = matrix.shape[1]
    for i in range(col_count):
        col_unique = np.unique(matrix[:, i])
        if (1 not in col_unique) or (-1 not in col_unique):
            return False
    return True


def _get_data_subset(data, label, target_label):
    """
    to get data with certain labels
    :param data: data set
    :param label: label corresponding to data
    :param target_label: the label which we want to get certain data
    :return:
    """
    data_subset = np.array([data[i] for i in range(len(label)) if label[i] in target_label])
    label_subset = np.array([label[i] for i in range(len(label)) if label[i] in target_label])
    return data_subset, label_subset


def _get_subset_feature_from_matrix(matrix, index):
    """
    I forget what it uses to.
    :param matrix:
    :param index:
    :return:
    """
    res = []
    for i in range(matrix.shape[1]):
        class_1 = []
        class_2 = []
        for j in range(matrix.shape[0]):
            if matrix[j, i] > 0:
                class_1.append(_get_key(index, j))
            elif matrix[j, i] < 0:
                class_2.append(_get_key(index, j))
        res.append(class_1)
        res.append(class_2)
    return res


def _create_confusion_matrix(y_true, y_pred, index):
    """
    create a confusion matrix
    :param y_true: true label
    :param y_pred: predicted label
    :param index: matrix index
    :return: confusion matrix
    """
    res = np.zeros((len(index), len(index)))
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            try:
                res[index[y_true[i]], index[y_pred[i]]] += 1
            except KeyError:
                pass
    return res + res.T


def _have_same_col(col, matrix):
    """
    to checkout wheather the col in coding matrix
    :param col:certain col to checkout
    :param matrix:coding matrix
    :return:true or false
    """
    col = col.reshape((1, -1))
    for i in range(matrix.shape[1]):
        if np.all([col == matrix[:, i]]) or np.all([col == -matrix[:, i]]):
            return True
    return False


def _create_col_from_partition(class_1_variety, class_2_variety, index):
    """
    create a col based on a certain partition
    :param class_1_variety: a part of partition as positive group
    :param class_2_variety: another part of partition as negative group
    :param index: index of coding matrix
    :return: a col
    """
    col = np.zeros((len(index), 1))
    for i in class_1_variety:
        col[index[i]] = 1
    for i in class_2_variety:
        col[index[i]] = -1
    return col


def _estimate_weight(error):
    """
    to estimate weights for base estimators based on the error rates
    :param error: error rates
    :return:
    """
    if error < 0.0000001:
        error = 0.0000001
    if error == 1:
        error = 0.9999999
    return 0.5*np.log((1-error)/error)

def logistic(x):
    return 1 / (1 + math.exp(-x))

def data_split(data_X, data_Y, test_size):
    """
    splite data into training set and testing set, dividing each class of data in a given proportion
    
    Parameters:
        data_X: data
        data_Y: labels
        test_size: the proportion of testing set in raw data
    Returns: 
        train_X, test_X, train_Y, test_Y
    """
    labels = np.unique(data_Y)
    temp_X = np.array([None] * len(labels))
    temp_Y = np.array([None] * len(labels))
    for i in range(len(labels)):
        label_i = (data_Y == labels[i])
        temp_X[i] = data_X[label_i]
        permutation = np.random.permutation(temp_X[i].shape[0])
        temp_X[i] = temp_X[i][permutation]
        temp_Y[i] = (data_Y[label_i])[permutation]

    test_X = np.concatenate( [temp_X[i][:int(len(temp_Y[i])*test_size)] for i in range(len(labels))] )
    test_Y = np.concatenate( [temp_Y[i][:int(len(temp_Y[i])*test_size)] for i in range(len(labels))] )
    train_X = np.concatenate( [temp_X[i][int(len(temp_Y[i])*test_size):] for i in range(len(labels))] )
    train_Y = np.concatenate( [temp_Y[i][int(len(temp_Y[i])*test_size):] for i in range(len(labels))] )
        
    return train_X, test_X, train_Y, test_Y