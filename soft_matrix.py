# soft coding tools
# 2018/3/10
# by HongZhou Guo

import numpy as np
import copy
import math

def _soften_matrix(coding_matrix, predict_matrix, labels):
    """
    soften the given coding matrix using soft values based on predict_matrix and labels

    Parameters:
        coding_matrix: the original harding coding matrix to be softened 
        predict_matrix: the matrix combinig predicted vector of the base leaners on each sample
        labels: the correspond label of the samples
    Returns: 
        soft value coding matrix
    """
    new_matrx = copy.deepcopy(coding_matrix)
    predicted = np.array(predict_matrix)
    for i in range(coding_matrix.shape[0]):
        new_matrx[i], std  = _get_mean_std(predicted[labels == i], False, coding_matrix[i])
    return new_matrx

def _interval_matrix(coding_matrix, predict_matrix, labels):
    """
    soften the given coding matrix using intervals ased on predict_matrix and labels

    Parameters:
        coding_matrix: the original harding coding matrix to be softened 
        predict_matrix: the matrix combinig predicted vector of the base leaners on each sample
        labels: the correspond label of the samples
    Returns: 
        soft interval coding matrix
    """
    new_matrix = np.array([None] * len(coding_matrix))
    predicted = np.array(predict_matrix)
    for i in range(coding_matrix.shape[0]):
        new_matrix[i] = np.array([None] * len(coding_matrix[i]))
        corr_res = predicted[labels == i]
        corr_mean, corr_std = _get_mean_std(corr_res, False, coding_matrix[i])
        for j in range(len(coding_matrix[i])):
            new_matrix[i][j] = {}
            new_matrix[i][j]['lower'] = corr_mean[j] - corr_std[j]
            new_matrix[i][j]['upper'] = corr_mean[j] + corr_std[j]
    return new_matrix

def _get_mean_std(predict_matrix, remove_singular=False, coding_vector=None):
    """
    calculate the mean value and the standard deviation of given predict matrix of current class
    
    Parameters:
        predict_matrix: the matrix combinig predicted vector of the base leaners on each sample
        remove_singular: boolean. if set True, function will remove singular point
        coding_vector: the associated codeword of current class
    Returns: 
        mean value and standard deviation of current class
    """
    mean_vector = np.mean(predict_matrix, axis=0)
    std_vector = np.std(predict_matrix, axis=0)
    if remove_singular == True:
        for i in range(len(predict_matrix[0])):
            temp_vector =  (predict_matrix[:,i])[[ (num*coding_vector[i] > 0) for num in predict_matrix[:,i] ]]
            mean_vector[i] = np.mean(temp_vector)
            std_vector[i] = np.std(temp_vector)
    return mean_vector, std_vector

def _weight_vector(coding_matrix, predict_matrix, labels):
    """
    calculate the weight for each base leaner
    
    Parameters:
        coding_matrix: the original harding coding matrix to be softened 
        predict_matrix: the matrix combinig predicted vector of the base leaners on each sample
        labels: the correspond label of the samples
    Returns: 
        weight_vector, the vector of weight of the base leaners
    """
    new_matrix = np.zeros((coding_matrix.shape[0], coding_matrix.shape[1]))
    predicted = np.array(predict_matrix)
    for i in range(coding_matrix.shape[0]):
        corr_res = predicted[labels == i]
        corr_mean, corr_std = _get_mean_std(corr_res, False, coding_matrix[i])
        for j in range(len(coding_matrix[i])):
            new_matrix[i][j] = 1 / math.exp(corr_std[j])
    weight_vector = np.mean(new_matrix, axis=0)
    return weight_vector

def _vector_score_interval(vector, interval_matrix, weight_vector, soft_matrix):
    """
    calculate the score of the output vector to each class (using soft value interval method)
    
    Parameters:
        vector: output vector of the base leaners
        interval_matrix: the soft interval matrix
        weight_vector: the weight vector of each base leaners
        soft_matrix: the soft (mean) values matrix
    Returns: 
        classifier result, the class index which has the highest score
    """
    index = None
    scores = [0] * interval_matrix.shape[0]
    for i in range(interval_matrix.shape[0]):
        for j in range(len(interval_matrix[i])):
            if vector[j] >= interval_matrix[i][j]['lower'] and vector[j] <= interval_matrix[i][j]['upper']:
                scores[i] += 1 * weight_vector[j] * math.exp(abs(vector[j])-1)
            else:
                distance = abs(vector[j] - soft_matrix[i][j])
                try:
                    scores[i] += math.exp(-distance) * weight_vector[j] * math.exp(abs(vector[j])-1)
                except OverflowError:
                    scores[i] = 0
        index = np.argmax(scores)
    return index

def _vector_score_soft(vector, soft_matrix):
    """
    calculate the minimum distance of the output vector to each class (using soft value method)
    
    Parameters:
        vector: output vector of the base leaners
        soft_matrix: the soft (mean) values matrix
    Returns: 
        classifier result, the class index which has the minimum distance
    """
    distances = [0] * soft_matrix.shape[0]
    for i in range(soft_matrix.shape[0]):
        for j in range(len(soft_matrix[i])):
            distances[i] += (vector[j] - soft_matrix[i][j]) ** 2
        index = np.argmin(distances)
    return index