"""
There are some common ECOC classifiers in this model, which shown as below
1.OVA ECOC
2.OVO ECOC
3.Dense random ECOC
4.Sparse random ECOC
5.D ECOC
6.AGG ECOC
7.CL_ECOC
8.ECOC_ONE

There are all defined as class, which inherit __BaseECOC

edited by Feng Kaijie
modified by HongZhou Guo
2017/11/30
"""

from abc import ABCMeta
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from Matrix_tool import _get_data_subset
from Matrix_tool import _closet_vector
from Matrix_tool import _exist_same_col
from Matrix_tool import _exist_same_row
from Matrix_tool import _exist_two_class
from Matrix_tool import _get_data_from_col
from Matrix_tool import _get_key
from Matrix_tool import _create_confusion_matrix
from Matrix_tool import _get_subset_feature_from_matrix
from Matrix_tool import _have_same_col
from Matrix_tool import _create_col_from_partition
from Matrix_tool import _estimate_weight
from Matrix_tool import _transform
from scipy.special import comb
from itertools import combinations
from Distance import euclidean_distance
from Distance import y_euclidean_distance
import Criterion
from SFFS import sffs
import copy

from soft_matrix import _soften_matrix
from soft_matrix import _interval_matrix
from soft_matrix import  _vector_score_soft
from soft_matrix import _vector_score_interval
from soft_matrix import  _weight_vector


from scipy.stats import kstest

__all__ = ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC', 'AGG_ECOC', 'CL_ECOC']


class __BaseECOC(object):
    """
    the base class for all to inherit
    """
    __metaclass__ = ABCMeta

    def __init__(self, distance_measure=euclidean_distance, base_estimator=svm.SVC, soften=False, soft_value=False):
        """
        :param distance_measure: a callable object to define the way to calculate the distance between predicted vector
                                    and true vector
        :param base_estimator: a class with fit and predict method, which define the base classifier for ECOC
            soften: Decide whether to soften the coding natrix, boolean. If true, use soft coding method.
            soft_value: Decide whether to use soft value methods, boolean. Only works when parameter 'soften' is true. 
        If true, use the soft value coding method. If false, use the interval coding method.
        """
        self.estimator = base_estimator
        self.predictors = []
        self.matrix = None
        self.index = None
        self.distance_measure = distance_measure
        self.weight_vector = None
        self.interval_matrix = None
        self.soft_matrix = None
        self.soften = soften
        self.soft_value = soft_value

    def create_matrix(self, data, label):
        """
        a method to create coding matrix for ECOC
        :param data: the data used in ecoc
        :param label: the corresponding label to data
        :return: coding matrix
        """
        raise AttributeError('create_matrix is not defined!')

    def fit(self, data, label, **estimator_param):
        """
        a method to train base estimator based on given data and label
        :param data: data used to train base estimator
        :param label: label corresponding to the data
        :param estimator_param: some param used by base estimator
        :return: None
        """
        self.predictors = []
        self.matrix, self.index = self.create_matrix(data, label)
        for i in range(self.matrix.shape[1]):
            dat, cla = _get_data_from_col(data, label, self.matrix[:, i], self.index)
            estimator = copy.deepcopy(self.estimator)
            estimator.fit(dat, cla)
            self.predictors.append(estimator)

    def predict(self, data, label):
        """
        a method used to predict label for give data
        :param data: data to predict
        :return: predicted label
        """
        res = []
        if len(self.predictors) == 0:
            raise ValueError('The Model has not been fitted!')
        if len(data.shape) == 1:
            data = np.reshape(data, [1, -1])
        count = 0
        predict_matrix = []
        for i in data:
            predicted_vector = self._use_predictors_proba(i)
            # predicted_vector = self._use_predictors(i)
            predict_matrix.append(predicted_vector)
            if self.soften == True:
                if self.soft_value == True:
                    value = _vector_score_soft(predicted_vector, self.soft_matrix)
                    
                else:
                    value = _vector_score_interval(predicted_vector, self.interval_matrix, self.weight_vector, self.soft_matrix)
            else:
                value = _closet_vector(predicted_vector, self.matrix)
            res.append(_get_key(self.index, value))
            count += 1
        return np.array(res)

    def predict_multi_return(self, data):
        if len(self.predictors) == 0:
            raise ValueError('The Model has not been fitted!')
        if len(data.shape) == 1:
            data = np.reshape(data, [1, -1])
        res = [0] * len(data)
        min_distance_vector = [0] * len(data)
        predicted_matrix = [None] * len(data)
        for i in range(len(data)):
            predicted_matrix[i] = self._use_predictors_proba(data[i])
            min_distance = np.inf
            for j in range(self.matrix.shape[0]):
                new_d = self.distance_measure(predicted_matrix[i], self.matrix[j])
                if new_d < min_distance:
                    min_distance = new_d
                    index = j
            min_distance_vector[i] = min_distance
            res[i] = _get_key(self.index, index)
        return np.array(res), min_distance_vector, predicted_matrix

    def validate(self, data, label, k=3, **estimator_params):
        res, min_distance, predict_matrix = self.predict_multi_return(data)
        values = _transform(label, self.index)
        self.weight_vector = _weight_vector(self.matrix, predict_matrix, values)
        self.interval_matrix = _interval_matrix(self.matrix, predict_matrix, values)
        self.soft_matrix = _soften_matrix(self.matrix, predict_matrix, values)

        """
        using cross validate method to validate model
        :param data: data used in validate
        :param label: ;label corresponding to data
        :param k: k-fold validate
        :param estimator_params: params used by base estimators
        :return: accuracy
        """
        # acc_list = []
        # kf = KFold(n_splits=k, shuffle=True)
        # original_predictors = copy.deepcopy(self.predictors)
        # for train_index, test_index in kf.split(data):
        #     data_train, data_test = data[train_index], data[test_index]
        #     label_train, label_test = label[train_index], label[test_index]
        #     self.fit(data_train, label_train, **estimator_params)
        #     label_predicted = self.predict(data_test)
        #     acc_list.append(accuracy_score(label_test, label_predicted))
        # self.predictors = copy.deepcopy(original_predictors)
        # return np.mean(acc_list)

    def _use_predictors(self, data):
        """
        :param data: data to predict
        :return: predicted vector
        """
        res = []
        for i in self.predictors:
            res.append(i.predict(np.array([data]))[0])
        return np.array(res)

    def _use_predictors_proba(self, data):
        """
        :param data: data to predict
        :return: predicted vector (probability)
        """
        res = []
        for i in self.predictors:
            temp = i.predict_proba(np.array([data]))[0]
            k = np.argmax(temp)
            res.append(i.classes_[k] * (temp[k] - 0.5) * 2)
        return np.array(res)


class OVA_ECOC(__BaseECOC):
    """
    ONE-VERSUS-ONE ECOC
    """

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = np.eye(len(index)) * 2 - 1
        return matrix, index


class OVO_ECOC(__BaseECOC):
    """
    ONE-VERSUS-ONE ECOC
    """

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        groups = combinations(range(len(index)), 2)
        matrix_row = len(index)
        matrix_col = np.int(comb(len(index), 2))
        col_count = 0
        matrix = np.zeros((matrix_row, matrix_col))
        for group in groups:
            class_1_index = group[0]
            class_2_index = group[1]
            matrix[class_1_index, col_count] = 1
            matrix[class_2_index, col_count] = -1
            col_count += 1
        return matrix, index


class Dense_random_ECOC(__BaseECOC):
    """
    Dense random ECOC
    """

    def create_matrix(self, data, label):
        while True:
            index = {l: i for i, l in enumerate(np.unique(label))}
            matrix_row = len(index)
            if matrix_row > 3:
                matrix_col = np.int(np.floor(10 * np.log10(matrix_row)))
            else:
                matrix_col = matrix_row
            matrix = np.random.random((matrix_row, matrix_col))
            class_1_index = matrix > 0.5
            class_2_index = matrix < 0.5
            matrix[class_1_index] = 1
            matrix[class_2_index] = -1
            if (not _exist_same_col(matrix)) and (not _exist_same_row(matrix)) and _exist_two_class(matrix):
                return matrix, index


class Sparse_random_ECOC(__BaseECOC):
    """
    Sparse random ECOC
    """
    def create_matrix(self, data, label):
        while True:
            index = {l: i for i, l in enumerate(np.unique(label))}
            matrix_row = len(index)
            if matrix_row > 3:
                matrix_col = np.int(np.floor(15 * np.log10(matrix_row)))
            else:
                matrix_col = np.int(np.floor(10 * np.log10(matrix_row)))
            matrix = np.random.random((matrix_row, matrix_col))
            class_0_index = np.logical_and(0.25 <= matrix, matrix < 0.75)
            class_1_index = matrix >= 0.75
            class_2_index = matrix < 0.25
            matrix[class_0_index] = 0
            matrix[class_1_index] = 1
            matrix[class_2_index] = -1
            if (not _exist_same_col(matrix)) and (not _exist_same_row(matrix)) and _exist_two_class(matrix):
                return matrix, index


class D_ECOC(__BaseECOC):
    """
    Discriminant ECOC
    """
    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = None
        labels_to_divide = [np.unique(label)]
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)
            datas, labels = _get_data_subset(data, label, label_set)
            class_1_variety_result, class_2_variety_result = sffs(datas, labels)
            new_col = np.zeros((len(index), 1))
            for i in class_1_variety_result:
                new_col[index[i]] = 1
            for i in class_2_variety_result:
                new_col[index[i]] = -1
            if matrix is None:
                matrix = copy.copy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            if len(class_1_variety_result) > 1:
                labels_to_divide.append(class_1_variety_result)
            if len(class_2_variety_result) > 1:
                labels_to_divide.append(class_2_variety_result)
        return matrix, index


class AGG_ECOC(__BaseECOC):
    """
    Agglomerative ECOC
    """
    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = None
        labels_to_agg = np.unique(label)
        labels_to_agg_list = [[x] for x in labels_to_agg]
        label_dict = {labels_to_agg[value]: value for value in range(labels_to_agg.shape[0])}
        num_of_length = len(labels_to_agg_list)
        class_1_variety = []
        class_2_variety = []
        while len(labels_to_agg_list) > 1:
            score_result = np.inf
            for i in range(0, len(labels_to_agg_list) - 1):
                for j in range(i + 1, len(labels_to_agg_list)):
                    class_1_data, class_1_label = _get_data_subset(data, label, labels_to_agg_list[i])
                    class_2_data, class_2_label = _get_data_subset(data, label, labels_to_agg_list[j])
                    score = Criterion.agg_score(class_1_data, class_1_label, class_2_data, class_2_label, score=Criterion.max_distance_score)
                    if score < score_result:
                        score_result = score
                        class_1_variety = labels_to_agg_list[i]
                        class_2_variety = labels_to_agg_list[j]
            new_col = np.zeros((num_of_length, 1))
            for i in class_1_variety:
                new_col[label_dict[i]] = 1
            for i in class_2_variety:
                new_col[label_dict[i]] = -1
            if matrix is None:
                matrix = new_col
            else:
                matrix = np.hstack((matrix, new_col))
            new_class = class_1_variety + class_2_variety
            labels_to_agg_list.remove(class_1_variety)
            labels_to_agg_list.remove(class_2_variety)
            labels_to_agg_list.insert(0, new_class)
        return matrix, index


class CL_ECOC(__BaseECOC):
    """
    Centroid loss ECOC, which use regressors as base estimators
    """
    def __init__(self, distance_measure=euclidean_distance, base_estimator=svm.SVR):
        super(CL_ECOC, self).__init__(distance_measure, base_estimator)

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = None
        labels_to_divide = [np.unique(label)]
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)
            datas, labels = _get_data_subset(data, label, label_set)
            class_1_variety_result, class_2_variety_result = sffs(datas, labels, score=Criterion.max_center_distance_score)
            class_1_data_result, class_1_label_result = _get_data_subset(data, label, class_1_variety_result)
            class_2_data_result, class_2_label_result = _get_data_subset(data, label, class_2_variety_result)
            class_1_center_result = np.average(class_1_data_result, axis=0)
            class_2_center_result = np.average(class_2_data_result, axis=0)
            belong_to_class_1 = [
                euclidean_distance(x, class_1_center_result) <= euclidean_distance(x, class_2_center_result)
                for x in class_1_data_result]
            belong_to_class_2 = [
                euclidean_distance(x, class_2_center_result) <= euclidean_distance(x, class_1_center_result)
                for x in class_2_data_result]
            class_1_true_num = {k: 0 for k in class_1_variety_result}
            class_2_true_num = {k: 0 for k in class_2_variety_result}
            for y in class_1_label_result[belong_to_class_1]:
                class_1_true_num[y] += 1
            for y in class_2_label_result[belong_to_class_2]:
                class_2_true_num[y] += 1
            class_1_label_count = {k: list(class_1_label_result).count(k) for k in class_1_variety_result}
            class_2_label_count = {k: list(class_2_label_result).count(k) for k in class_2_variety_result}
            class_1_ratio = {k: class_1_true_num[k] / class_1_label_count[k] for k in class_1_variety_result}
            class_2_ratio = {k: -class_2_true_num[k] / class_2_label_count[k] for k in class_2_variety_result}
            new_col = np.zeros((len(index), 1))
            for i in class_1_ratio:
                new_col[index[i]] = class_1_ratio[i]
            for i in class_2_ratio:
                new_col[index[i]] = class_2_ratio[i]
            if matrix is None:
                matrix = copy.copy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            if len(class_1_variety_result) > 1:
                labels_to_divide.append(class_1_variety_result)
            if len(class_2_variety_result) > 1:
                labels_to_divide.append(class_2_variety_result)
        return matrix, index


class ECOC_ONE(__BaseECOC):
    """
    ECOC-ONE:Optimal node embedded ECOC
    """
    def __init__(self, distance_measure=euclidean_distance, base_estimator=svm.SVC, iter_num=10, **param):
        self.train_data = None
        self.validate_data = None
        self.train_label = None
        self.validation_y = None
        self.estimator = base_estimator
        self.matrix = None
        self.index = None
        self.predictors = None
        self.predictor_weights = None
        self.iter_num = iter_num
        self.param = param
        self.distance_measure = distance_measure

    def create_matrix(self, train_data, train_label, validate_data, validate_label, estimator, **param):
        index = {l: i for i, l in enumerate(np.unique(train_label))}
        matrix = None
        predictors = []
        predictor_weights = []
        labels_to_divide = [np.unique(train_label)]
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)
            label_count = len(label_set)
            groups = combinations(range(label_count), np.int(np.ceil(label_count / 2)))
            score_result = 0
            est_result = None
            for group in groups:
                class_1_variety = np.array([label_set[i] for i in group])
                class_2_variety = np.array([l for l in label_set if l not in class_1_variety])
                class_1_data, class_1_label = _get_data_subset(train_data, train_label, class_1_variety)
                class_2_data, class_2_label = _get_data_subset(train_data, train_label, class_2_variety)
                class_1_cla = np.ones(len(class_1_data))
                class_2_cla = -np.ones(len(class_2_data))
                train_d = np.vstack((class_1_data, class_2_data))
                train_c = np.hstack((class_1_cla, class_2_cla))
                est = estimator(**param).fit(train_d, train_c)
                class_1_data, class_1_label = _get_data_subset(validate_data, validate_label, class_1_variety)
                class_2_data, class_2_label = _get_data_subset(validate_data, validate_label, class_2_variety)
                class_1_cla = np.ones(len(class_1_data))
                class_2_cla = -np.ones(len(class_2_data))
                validation_d = np.array([])
                validation_c = np.array([])
                try:
                    validation_d = np.vstack((class_1_data, class_2_data))
                    validation_c = np.hstack((class_1_cla, class_2_cla))
                except Exception:
                    if len(class_1_data) > 0:
                        validation_d = class_1_data
                        validation_c = class_1_cla
                    elif len(class_2_data) > 0:
                        validation_d = class_2_data
                        validation_c = class_2_cla
                if validation_d.shape[0] > 0 and validation_d.shape[1] > 0:
                    score = est.score(validation_d, validation_c)
                else:
                    score = 0.8
                if score >= score_result:
                    score_result = score
                    est_result = est
                    class_1_variety_result = class_1_variety
                    class_2_variety_result = class_2_variety
            new_col = np.zeros((len(index), 1))
            for i in class_1_variety_result:
                new_col[index[i]] = 1
            for i in class_2_variety_result:
                new_col[index[i]] = -1
            if matrix is None:
                matrix = copy.copy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            predictors.append(est_result)
            predictor_weights.append(_estimate_weight(1 - score_result))
            if len(class_1_variety_result) > 1:
                labels_to_divide.append(class_1_variety_result)
            if len(class_2_variety_result) > 1:
                labels_to_divide.append(class_2_variety_result)
        return matrix, index, predictors, predictor_weights

    def fit(self, data, label):
        self.train_data, self.validate_data, self.train_label, self.validation_y = train_test_split(data, label,
                                                                                                    test_size=0.25)
        self.matrix, self.index, self.predictors, self.predictor_weights = \
            self.create_matrix(self.train_data, self.train_label, self.validate_data, self.validation_y, self.estimator, **self.param)
        feature_subset = _get_subset_feature_from_matrix(self.matrix, self.index)
        for i in range(self.iter_num):
            y_pred = self.predict(self.validate_data)
            y_true = self.validation_y
            confusion_matrix = _create_confusion_matrix(y_true, y_pred, self.index)
            while True:
                max_index = np.argmax(confusion_matrix)
                max_index_y = np.floor(max_index / confusion_matrix.shape[1])
                max_index_x = max_index % confusion_matrix.shape[1]
                label_y = _get_key(self.index, max_index_y)
                label_x = _get_key(self.index, max_index_x)
                score_result = 0
                col_result = None
                est_result = None
                est_weight_result = None
                feature_subset_m = None
                feature_subset_n = None
                for m in range(len(feature_subset)-1):
                    for n in range(m+1, len(feature_subset)):
                        if ((label_y in feature_subset[m] and label_x in feature_subset[n])
                                or (label_y in feature_subset[n] and label_x in feature_subset[m])) \
                                and (set(feature_subset[m]).intersection(set(feature_subset[n])) == set()):
                            col = _create_col_from_partition(feature_subset[m], feature_subset[n], self.index)
                            if not _have_same_col(col, self.matrix):
                                train_data, train_cla = _get_data_from_col(self.train_data, self.train_label, col, self.index)
                                est = self.estimator(**self.param).fit(train_data, train_cla)
                                validation_data, validation_cla = _get_data_from_col(self.validate_data, self.validation_y, col, self.index)
                                if validation_data is None:
                                    score = 0.8
                                else:
                                    score = est.score(validation_data, validation_cla)
                                if score >= score_result:
                                    score_result = score
                                    col_result = col
                                    est_result = est
                                    est_weight_result = _estimate_weight(1-score_result)
                                    feature_subset_m = m
                                    feature_subset_n = n
                if col_result is None:
                    confusion_matrix[np.int(max_index_y), np.int(max_index_x)] = 0
                    if np.sum(confusion_matrix) == 0:
                        break
                else:
                    break
            try:
                self.matrix = np.hstack((self.matrix, col_result))
                self.predictors.append(est_result)
                self.predictor_weights.append(est_weight_result)
                feature_subset.append(feature_subset[feature_subset_m] + feature_subset[feature_subset_n])
            except (TypeError, ValueError):
                pass

    def predict(self, data):
        res = []
        if len(self.predictors) == 0:
            raise ValueError('The Model has not been fitted!')
        if len(data.shape) == 1:
            data = np.reshape(data, [1, -1])
        for i in data:
            predict_res = self._use_predictors(i)
            value = _closet_vector(predict_res, self.matrix, y_euclidean_distance, np.array(self.predictor_weights))
            res.append(_get_key(self.index, value))
        return np.array(res)