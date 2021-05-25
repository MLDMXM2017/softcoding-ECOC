# test accuracy
# 2017/11/30
# by HongZhou Guo

# -*- coding: utf-8 -*- 

import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, tree, svm, neighbors
from sklearn.linear_model import Lasso
from sklearn import preprocessing
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer
from Matrix_tool import _transform, data_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import random
import math
import Classifier
import soft_matrix
import tool
from sklearn import metrics

from sklearn.gaussian_process import GaussianProcessClassifier

# test the datasets under this directory
path = 'DataSet\\test_set'
file_list = []
for root, dirs, files in os.walk(path):  
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
            file_list.append(file)
print(file_list)

file_outpath = tool.mkdir('softcoding_report\\' + tool.get_time_str())
out = open(file_outpath + '\\accuracy.csv','w')
out2 = open(file_outpath + '\\matrix.csv','w')

# import sys
# savedStdout = sys.stdout
# sys.stdout = out

SEPERATOR = '\n--------------------------------------------------------------------------\n'
REPEAT_TIMES = 10

for file_num in range(len(file_list)):
    print(file_list[file_num])
    out.writelines(file_list[file_num])
    out.writelines('\n')
    data_set = np.loadtxt(os.path.join(path, file_list[file_num]), delimiter=',', dtype=bytes)
    data_X = data_set[:,:-1].astype('float64')
    data_Y = np.ravel(data_set[:,-1:].astype('str'))
    data_X = Imputer().fit_transform(data_X)
    filename = os.path.splitext(file_list[file_num])[0]
    data_X = preprocessing.StandardScaler().fit_transform(data_X, data_Y)

    accuracy_soft_interval = [0] * REPEAT_TIMES
    accuracy_soft_value = [0] * REPEAT_TIMES
    accuracy_hard = [0] * REPEAT_TIMES
    precision_soft_interval = [0] * REPEAT_TIMES
    precision_soft_value = [0] * REPEAT_TIMES
    precision_hard = [0] * REPEAT_TIMES
    recall_soft_interval = [0] * REPEAT_TIMES
    recall_soft_value = [0] * REPEAT_TIMES
    recall_hard = [0] * REPEAT_TIMES
    f1score_soft_interval = [0] * REPEAT_TIMES
    f1score_soft_value = [0] * REPEAT_TIMES
    f1score_hard = [0] * REPEAT_TIMES

    print(file_list[file_num])

    for t in range(REPEAT_TIMES):
        
        train_X, test_X, train_Y, test_Y = data_split(data_X, data_Y, test_size=0.5)
        validate_X, test_X, validate_Y, test_Y = data_split(test_X, test_Y, test_size=0.5)

        # base_leaner = GaussianProcessClassifier()  
        # base_leaner = RandomForestClassifier()
        # base_leaner = MLPClassifier()
        base_leaner = svm.SVC(kernel='rbf', probability=True)

        clf_soft_interval = Classifier.AGG_ECOC(base_estimator=base_leaner, soften=True)
        clf_soft_value = Classifier.AGG_ECOC(base_estimator=base_leaner, soften=True, soft_value=True)
        clf_hard = Classifier.AGG_ECOC(base_estimator=base_leaner)

        clf_soft_interval.fit(train_X, train_Y)
        clf_soft_interval.validate(validate_X, validate_Y)
        res = clf_soft_interval.predict(test_X, np.ravel(test_Y))
        accuracy_soft_interval[t] = metrics.precision_score(test_Y, res, average='micro')
        precision_soft_interval[t] = metrics.precision_score(test_Y, res, average='macro')
        recall_soft_interval[t] = metrics.recall_score (test_Y, res, average='macro')
        f1score_soft_interval[t] = metrics.f1_score (test_Y, res, average='macro')
        print("=========" + str(t) + "========")
        print("soft interval")
        print(metrics.classification_report(test_Y, res))

        # print('accuracy report: \n' + metrics.classification.classification_report(test_Y, res))

        clf_soft_value.fit(train_X, train_Y)
        clf_soft_value.validate(validate_X, validate_Y)
        res2 = clf_soft_value.predict(test_X, np.ravel(test_Y))
        accuracy_soft_value[t] = metrics.precision_score(test_Y, res2, average='micro')
        precision_soft_value[t] = metrics.precision_score(test_Y, res2, average='macro')
        recall_soft_value[t] = metrics.recall_score(test_Y, res2, average='macro')
        f1score_soft_value[t] = metrics.f1_score(test_Y, res2, average='macro')
        print("soft value")
        print(metrics.classification_report(test_Y, res2))
        # print('accuracy report: \n' + metrics.classification.classification_report(test_Y, res2))
        
        clf_hard.fit(train_X, train_Y)
        res_hard = clf_hard.predict(test_X, np.ravel(test_Y))
        accuracy_hard[t] = metrics.precision_score(test_Y, res_hard, average='micro')
        precision_hard[t] = metrics.precision_score(test_Y, res_hard, average='macro')
        recall_hard[t] = metrics.recall_score(test_Y, res_hard, average='macro')
        f1score_hard[t] = metrics.f1_score(test_Y, res_hard, average='macro')
        print("hard")
        print(metrics.classification_report(test_Y, res_hard))
        # print('accuracy report: \n' + metrics.classification.classification_report(test_Y, res_hard))
        
        matrix = clf_soft_value.matrix
        out2.writelines('hard coding matrix:\n')
        for line in matrix:
            out2.writelines(','.join([ str(round(data,2)) for data in line ]))
            out2.writelines('\n')
        out2.writelines('\n')
        out2.writelines('soft value coding matrix:\n')
        matrix = clf_soft_interval.soft_matrix
        for line in matrix:
            out2.writelines(','.join([ str(data) for data in line ]))
            out2.writelines('\n')
        out2.writelines('\n')
        out2.writelines('interval coding matrix:\n')
        matrix = clf_soft_interval.interval_matrix
        for line in matrix:
            out2.writelines(','.join([ str(data) for data in line ]))
            out2.writelines('\n')
        out2.writelines('\n')

    out.writelines(',hard coding, soft interval coding, soft value coding')
    out.writelines('\n')
    out.writelines('accuracy,' + str(np.mean(accuracy_hard)) + ',' + str(np.mean(accuracy_soft_interval)) + ','+ str(np.mean(accuracy_soft_value)))
    out.writelines('\n')
    out.writelines('precision,' + str(np.mean(precision_hard)) + ',' + str(np.mean(precision_soft_interval)) + ','+ str(np.mean(precision_soft_value)))
    out.writelines('\n')
    out.writelines('recall,' + str(np.mean(recall_hard)) + ',' + str(np.mean(recall_soft_interval)) + ','+ str(np.mean(recall_soft_value)))
    out.writelines('\n')
    out.writelines('f1-score,' + str(np.mean(f1score_hard)) + ',' + str(np.mean(f1score_soft_interval)) + ','+ str(np.mean(f1score_soft_value)))
    out.writelines('\n')

    out.writelines(SEPERATOR)
    out2.writelines(SEPERATOR)
    out.flush()
    out2.flush()
out.close()
out2.close()

fin = open('finish','w')
fin.close()